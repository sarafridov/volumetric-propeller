from cmath import pi
import numpy as np
import os

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)

gpu = get_freer_gpu()
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

import torch
import torch.nn as nn
import sigpy as sigpy
from tqdm import tqdm
import imageio
from PIL import Image
import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches

np.random.seed(0)
torch.random.seed()
dtype = torch.float32
device = torch.device("cuda:0")

class PropMRIOffResonanceCorrection(nn.Module):

  # __init__ function to initialize values input to model
  def __init__(self, dim_x, dim_y, dim_z, initial_val=0.01, blade_width=None, num_views=5, expname=''):
    super().__init__()

    ### Make the log dirs
    self.log_train = os.path.join(expname, 'train')
    self.log_test = os.path.join(expname, 'test')
    os.makedirs(self.log_train, exist_ok=True)
    os.makedirs(self.log_test, exist_ok=True)

    ### Dimensions
    self.dim_x, self.dim_y, self.dim_z = dim_x, dim_y, dim_z

    ### Create image for training
    image = torch.from_numpy(np.flip(sigpy.shepp_logan(shape=(dim_x, dim_y)), axis=0).copy())
    self.image = torch.real(image).to(dtype)

    self.image_fat_layer = torch.clone(self.image)
    self.image_fat_layer[self.image_fat_layer < 1] = 0
    self.image_water_layer = self.image - self.image_fat_layer
    self.image_fat_layer = self.image_fat_layer.to(device)
    self.image_water_layer = self.image_water_layer.to(device)

    vis = self.image.detach()
    vis = vis - torch.min(vis)
    vis = np.asarray((vis * 255).cpu()).astype(np.uint8)
    imageio.imwrite(os.path.join(self.log_train, 'gtimg.png'), vis)

    ### Initialize a model with real and imaginary components
    A = torch.ones(size=(dim_x, dim_y, dim_z), dtype=dtype, device=device)
    B = torch.ones(size=(dim_x, dim_y, dim_z), dtype=dtype, device=device)
    self.real_model = nn.Parameter(data=A * initial_val)
    self.imag_model = nn.Parameter(data=B * initial_val)

    ### Embed fat and water layers along Gaussian contours
    self.true_real = torch.zeros(size=(dim_x, dim_y, dim_z), dtype=dtype, device=device)
    self.true_imag = torch.zeros(size=(dim_x, dim_y, dim_z), dtype=dtype, device=device)  # For Shepp, true_imag is actually zeros
    
    def make_gaussian(x, y, N):
      alpha = 6*N
      y_scale = 4
      return torch.exp(-(1/(2*y_scale*alpha)) * (x - 7*N/8)**2 - (1/(2*alpha)) * (y - 5*N/9)**2).to(device)

    xs, ys = torch.meshgrid(torch.arange(dim_x), torch.arange(dim_y), indexing='ij')
    xs = xs.to(device)
    ys = ys.to(device)
    self.true_real[xs, ys, (dim_z/5 + dim_z / 4 * make_gaussian(xs, ys, dim_x)).long()] = self.image_water_layer
    self.true_real[xs, ys, (3*dim_z/4  + dim_z / 4 * make_gaussian(xs, ys, dim_x)).long()] = self.image_fat_layer

    ### Generate training and testing angles
    all_angles = np.arange(0, pi+0.01, pi/num_views)[1:]
    print(f'training with {len(all_angles)} views')
    np.random.shuffle(all_angles)
    self.training_angles = all_angles
    self.test_angle = pi/2 # This is somewhat arbitrary since it is overhead anyway

    ### Generate training and testing data => Truth values @ initialization
    self.training_data = []
    self.training_kspace = []
    for theta in self.training_angles:
      img_ren = self.render_img(theta, use_overhead = False, use_predicted = False)
      # Save copies of the raw projections for later use
      np.save(os.path.join(self.log_train, f'pre_blur_train_{theta}.npy'), np.array(img_ren.cpu()))
      gt_kspace, gt_img = self.blur_img(img=img_ren, theta=theta)
      self.training_data.append(gt_img)
      self.training_kspace.append(gt_kspace)
      # Save copies of the masks for visualization
      mask = self.make_mask(theta, img_ren.shape[0])
      vis = mask.detach().cpu()
      np.save(os.path.join(self.log_train, f'mask_{theta}.npy'), np.array(mask.cpu()))
      vis = np.asarray(vis*255).astype(np.uint8)
      imageio.imwrite(os.path.join(self.log_train, f'mask_{theta}.png'), vis)
    
    self.testing_data = self.render_img(self.test_angle, use_overhead = True, use_predicted = False)
    vis = torch.abs(self.testing_data.detach())
    vis = vis - torch.min(vis)
    vis = np.asarray((vis * 255).cpu()).astype(np.uint8)
    imageio.imwrite(os.path.join(self.log_test, 'overheadtruth.png'), vis)
  
  def make_mask(self, theta, N):
    im = Image.fromarray(np.uint8(np.zeros((N, N))))
    fig, ax = plt.subplots()
    _ = ax.imshow(im)
    width = N
    height = N * np.sqrt((3-np.sqrt(5.)) / (5 + np.sqrt(5.)))
    x = 0
    y = N/2-height / 2
    rect = patches.Rectangle((x, y), width=width, height=height, linewidth=0, angle=theta*180/np.pi, rotation_point='center', edgecolor='white', facecolor='white')
    ax.add_patch(rect)
    
    io_buf=io.BytesIO()
    fig.savefig(io_buf, format='raw')
    io_buf.seek(0)
    data = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()
    plt.close()

    # Crop out our image from the border
    ycrop = np.where(data[data.shape[0]//2,:,0]==0)[0] # y min
    xcrop = np.where(data[:,data.shape[1]//2,0]==0)[0] # x min
    cropped = data[xcrop[0]:xcrop[1], ycrop[0]:ycrop[1]].copy()

    # Make it black and white
    cropped[cropped < 200] = 0

    # Resize back to N by N
    im = Image.fromarray(cropped)
    im = im.resize((N, N))
    cropped = np.array(im)
    # Change data type
    cropped = np.array(cropped[:,:,0], dtype=np.float32) / 255
    mask = torch.from_numpy(cropped).to(device)

    return mask


  # Blur an image using a PROPELLER blade k-space mask
  # Expects input to be complex, and returns a complex image
  def blur_img(self, img, theta):
    assert img.dtype == torch.complex64
    mask = self.make_mask(theta, img.shape[0])
    img_FFT = torch.fft.fft2(img)
    img_DFT = torch.fft.fftshift(img_FFT)
    img_DFT_masked = img_DFT * mask
    inv_img_DFT_masked = torch.fft.ifftshift(img_DFT_masked)
    inv_img_DFT_masked = torch.fft.ifft2(inv_img_DFT_masked)
    return img_DFT_masked, inv_img_DFT_masked


  # Compute a projection of the volume at the measurement angle. Result is a 2D complex-valued image.
  def render_img(self, measurement_angle, use_overhead = False, use_predicted = True, which_layer='all'):
    xs, ys = torch.meshgrid(torch.arange(self.real_model.shape[0]), torch.arange(self.real_model.shape[1]), indexing='ij')
    image_real = self.render_image(xs, ys, measurement_angle, use_overhead, use_predicted, use_real=True, which_layer=which_layer)
    image_imag = self.render_image(xs, ys, measurement_angle, use_overhead, use_predicted, use_real=False, which_layer=which_layer)
    # Combine into a single complex image
    image = torch.complex(image_real, image_imag).to(device)
    assert image.dtype == torch.complex64
    return image


  def render_image(self, pixel_x, pixel_y, measurement_angle, use_overhead, use_predicted, use_real, which_layer='all'):
    ### pixel_x and pixel_y are 2D vectors with shape [W, H] the pixel indices we want to render in an image
    if use_predicted:
      if use_real:
        grid = self.real_model
      else:
        grid = self.imag_model
    else:
      if use_real:
        grid = self.true_real
      else:
        grid = self.true_imag

    ### get intersection points (x, y, z)
    xs, ys, zs, n_pts = self.get_intersection_pts(pixel_x, pixel_y, measurement_angle, use_overhead) # xs, ys, zs, masks should each have shape [W, H, npts]

    ### normalize, reshape and permute
    normalized_xs = (2*xs)/self.dim_x - 1
    normalized_ys = (2*ys)/self.dim_y - 1
    normalized_zs = (2*zs)/self.dim_z - 1

    xs = normalized_xs[None, None, ...]
    ys = normalized_ys[None, None, ...]
    zs = normalized_zs[None, None, ...]

    ### render just fat, just water, or all together
    if which_layer == 'fat':
      keep_idx = (zs >= 0)
      xs = xs[keep_idx].reshape(xs.shape[:-1] + (-1,))
      ys = ys[keep_idx].reshape(xs.shape[:-1] + (-1,))
      zs = zs[keep_idx].reshape(xs.shape[:-1] + (-1,))
      n_pts = zs.shape[-1]
    elif which_layer == 'water':
      keep_idx = (zs < 0)
      xs = xs[keep_idx].reshape(xs.shape[:-1] + (-1,))
      ys = ys[keep_idx].reshape(xs.shape[:-1] + (-1,))
      zs = zs[keep_idx].reshape(xs.shape[:-1] + (-1,))
      n_pts = zs.shape[-1]
    else:
      assert which_layer == 'all'


    xs_perm = torch.permute(xs, dims=(4, 1, 2, 3, 0))
    ys_perm = torch.permute(ys, dims=(4, 1, 2, 3, 0))
    zs_perm = torch.permute(zs, dims=(4, 1, 2, 3, 0))

    ### points
    input_grid = grid[None, None, ...]
    input_grid = input_grid.repeat(n_pts, 1, 1, 1, 1)

    grid_input = torch.cat((zs_perm, ys_perm, xs_perm), -1).cuda()
    points_grid_samp = torch.nn.functional.grid_sample(input_grid, grid_input, mode='bilinear', padding_mode='zeros', align_corners=False)
    points = torch.permute(points_grid_samp, dims=(3, 4, 0, 1, 2)).squeeze()

    pixel_value = torch.sum(points, dim=-1)
    return pixel_value


  # Return the points of intersection of the ray based on measurement angle and the pixels
  def get_intersection_pts(self, pixel_x, pixel_y, measurement_angle, use_overhead):
    ### pixel_x and pixel_y are shape [W, H]
    if use_overhead:
      offset_angle = 0.1/180*pi
    else:
      offset_angle = 20/180*pi # This is a design parameter that is somewhat linked to the z (omega) resolution
    

    step_size = 0.5
    n_pts = (int)(self.real_model.shape[2] * 2 / step_size) # upper bound
    x_step = step_size * np.cos(measurement_angle) * np.sin(offset_angle)
    y_step = step_size * np.sin(measurement_angle) * np.sin(offset_angle)
    z_step = step_size * np.cos(offset_angle)


    if (np.cos(measurement_angle) * np.sin(offset_angle) == 0):
      xs = torch.zeros(n_pts)[None,None,0:n_pts] + torch.zeros(pixel_x.shape[0], pixel_x.shape[1], 1)
    else:
      xs = torch.arange(start=0, end=0 + n_pts*x_step, step=x_step)[None,None,0:n_pts] + pixel_x[:,:,None]

    if (np.sin(measurement_angle) * np.sin(offset_angle) == 0):
      ys = torch.zeros(n_pts)[None,None,0:n_pts] + torch.zeros(pixel_x.shape[0], pixel_x.shape[1], 1)
    else:
      ys = torch.arange(start=0, end=0 + n_pts*y_step, step=y_step)[None,None,0:n_pts] + pixel_y[:,:,None]

    if (np.cos(offset_angle) == 0):
      zs = torch.zeros(n_pts)[None,None,0:n_pts] + torch.zeros(pixel_x.shape[0], pixel_x.shape[1], 1)
    else:
      zs = torch.arange(start=0, end=0 + n_pts*z_step, step=z_step)[None,None,0:n_pts] + torch.zeros(pixel_x.shape[0], pixel_x.shape[1], 1)

    ### xs, ys, zs should all have shape [W, H, npts]
    return xs, ys, zs, n_pts


  def compute_tv(self):
    # shift the model left, right, up, down, front, back and sum the absolute value differences
    model = torch.complex(self.real_model, self.imag_model).to(device)
    x_diff = model[1:,:,:] - model[:-1,:,:]
    y_diff = model[:,1:,:] - model[:,:-1,:]
    z_diff = model[:,:,1:] - model[:,:,:-1]
    return torch.mean(torch.abs(x_diff)) + torch.mean(torch.abs(y_diff)) + torch.mean(torch.abs(z_diff))


  def compute_loss(self, measurement_angle, gt_img, kspace_gt=None, use_overhead=False, tv_lambda=10):
    img_predicted = self.render_img(measurement_angle, use_overhead, use_predicted = True) # rendered
    
    if use_overhead:
      log_dir = self.log_test
      mse_loss = torch.nn.functional.mse_loss(torch.abs(img_predicted), torch.abs(gt_img))
      # Also save fat-only and water-only predictions
      img_fat = self.render_img(measurement_angle, use_overhead, use_predicted = True, which_layer='fat')
      img_water = self.render_img(measurement_angle, use_overhead, use_predicted = True, which_layer='water')
      vis = torch.abs(torch.cat((img_water, img_fat), dim=1)).detach()
      vis = vis - torch.min(vis)
      vis = np.asarray((vis/torch.max(vis) * 255).cpu()).astype(np.uint8)
      imageio.imwrite(os.path.join(log_dir, f'fat_water.png'), vis)
    else:
      assert kspace_gt is not None
      kspace_predicted, img_predicted = self.blur_img(img_predicted, measurement_angle) # rendered and blurred
      log_dir = self.log_train
      # Use mostly image-space loss, but a little bit of k-space loss
      mse_loss = 0.000001 * torch.mean(torch.abs(torch.square(kspace_predicted - kspace_gt))) + 0.999999 * torch.mean(torch.abs(torch.square(img_predicted - gt_img)))

    ### Visualize/imwrite img_blurred and gt_img_blurred
    if (log_dir != None):
      vis = torch.abs(torch.cat((gt_img, img_predicted), dim=1)).detach()
      vis = vis - torch.min(vis)
      vis = np.asarray((vis/torch.max(vis) * 255).cpu()).astype(np.uint8)
      imageio.imwrite(os.path.join(log_dir, f'angle_{measurement_angle}.png'), vis)
  
    assert mse_loss.dtype == torch.float32
    tv_loss = 0
    if tv_lambda > 0:
        tv_loss = tv_lambda * self.compute_tv()
    return mse_loss, tv_loss


  def save_model(self):
      np.save(os.path.join(self.log_train, f'gt_real.npy'), np.array(self.true_real.detach().cpu()))
      np.save(os.path.join(self.log_train, f'recon_real.npy'), np.array(self.real_model.detach().cpu()))
      np.save(os.path.join(self.log_train, f'gt_imag.npy'), np.array(self.true_imag.detach().cpu()))
      np.save(os.path.join(self.log_train, f'recon_imag.npy'), np.array(self.imag_model.detach().cpu()))


  def save_slices(self, slices):
    for sl in slices:
      gt = self.true_real[:,:,sl]
      pred = self.real_model[:,:,sl]
      vis = torch.cat((gt, pred), dim=1).detach()
      vis = vis - torch.min(vis)
      vis = np.asarray((vis/torch.max(vis) * 255).cpu()).astype(np.uint8)
      imageio.imwrite(os.path.join(self.log_test, f'slice_{sl}.png'), vis)


  def train(self, num_epochs, lr=0.0025, tv_lambda=0):
    best_overhead_psnr = 0.0

    optimizer = torch.optim.Adam(self.parameters(), lr=lr)
    pb = tqdm(total=num_epochs)

    for epoch in range(num_epochs):
        count = 0
        train_loss_total = 0

        for measurement_angle, train_image, train_kspace in tqdm(zip(self.training_angles, self.training_data, self.training_kspace)):
            optimizer.zero_grad() ### zero the parameter gradients

            ### compute the loss based on model output and the ground truth
            mse, tv = self.compute_loss(measurement_angle, gt_img=train_image, kspace_gt=train_kspace, use_overhead=False, tv_lambda=tv_lambda)
            loss = mse + tv
            train_loss_total += loss
            count += 1
            loss.backward() ### backpropagate the loss
            optimizer.step() ### adjust parameters based on the calculated gradients

        avg_train_loss = train_loss_total/count

        ### Compute and print the PSNR for this epoch when tested overhead
        mseloss, tvloss = self.compute_loss(self.test_angle, gt_img=self.testing_data, use_overhead=True, tv_lambda=tv_lambda)
        overhead_psnr = -10*torch.log10(mseloss)
        pb.set_postfix_str(f'Epoch {epoch+1}: train psnr={-10*torch.log10(avg_train_loss):.4f}, overhead psnr={overhead_psnr}, mseloss={mseloss}, tvloss={tvloss}', refresh=False)
        pb.update(1)
        
        # Save the model if this epoch accuracy is the best
        if overhead_psnr > best_overhead_psnr:
            self.save_model()
            self.save_slices(slices=[self.dim_z//5, self.dim_z//5 + 1, 3*self.dim_z//4, 3*self.dim_z//4 + 1])
            best_overhead_psnr = overhead_psnr

    pb.close()
    


if __name__ == "__main__":
  # Assumes 5 blades that exactly cover the space; blade width is computed automatically
  reso = 180
  tv_lambda = 2.0
  lr = 0.02 
  model = PropMRIOffResonanceCorrection(reso, reso, 30, initial_val=0.1, expname=f'shepp_reso{reso}_tv{tv_lambda}_lr{lr}_0.999999realspace')
  model.train(100, lr=lr, tv_lambda=tv_lambda)
  print(model.log_test)
