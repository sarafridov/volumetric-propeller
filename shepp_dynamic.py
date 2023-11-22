# imports
from cmath import pi
import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import gc
import logging as log

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)

gpu = get_freer_gpu()
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
print(f'using GPU {gpu}')

import torch
import torch.nn as nn
from tqdm import tqdm
import imageio
from PIL import Image
import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from k_planes_field import KPlaneField, TimeSmoothness, L1TimePlanes, PlaneTV
from typing import Optional, Union, List, Dict, Sequence, Iterable, Collection, Callable
from phantominator import ct_shepp_logan
from skimage.restoration import denoise_nl_means

np.random.seed(0)
torch.random.seed()
dtype = torch.float32
device = torch.device("cuda:0")

def write_video_to_file(file_name, frames: List[np.ndarray]):
  log.info(f"Saving video ({len(frames)} frames) to {file_name}")
  # Photo tourism image sizes differ
  sizes = np.array([frame.shape[:2] for frame in frames])
  same_size_frames = np.unique(sizes, axis=0).shape[0] == 1
  if same_size_frames:
      height, width = frames[0].shape[:2]
      video = cv2.VideoWriter(
          file_name, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height), False)
      frames_max_val = np.max(np.abs(np.array(torch.stack(frames).cpu())).flatten())
      for img in frames:
          if isinstance(img, torch.Tensor):
              img = img.cpu().numpy()
          img = (np.abs(img)*255/frames_max_val).astype(np.uint8)
          video.write(img)  
      cv2.destroyAllWindows()
      video.release()


class PropMRIOffResonanceCorrection(nn.Module):

  # __init__ function to initialize values input to model
  def __init__(self, dim_x, dim_y, dim_z, expname='', plane_tv_weight = 0, l1_time_planes = 0, time_smoothness_weight = 0):
    super().__init__()

    print(expname)

    ### Make the log dirs
    self.log_train = os.path.join(expname, 'train')
    self.log_test = os.path.join(expname, 'test')
    self.log_planes = os.path.join(expname, 'planes')
    os.makedirs(self.log_train, exist_ok=True)
    os.makedirs(self.log_test, exist_ok=True)
    os.makedirs(self.log_planes, exist_ok=True)

    ### Dimensions
    self.dim_x, self.dim_y, self.dim_z = dim_x, dim_y, dim_z

    ### Set number of training views
    self.num_train_views = 67
    self.test_angle = pi/2

    _, E = ct_shepp_logan((self.dim_x, self.dim_y), ret_E = True)
    volumes_list = []

    def make_gaussian(x, y, N):
        alpha = 6*N
        y_scale = 4
        return torch.exp(-(1/(2*y_scale*alpha)) * (x - 7*N/8)**2 - (1/(2*alpha)) * (y - 5*N/9)**2)
    
    timesteps = np.arange(0, self.num_train_views)

    # Create synthetic 3D dynamic data
    self.training_volumes = torch.zeros(size=(len(timesteps), self.dim_x, self.dim_y, self.dim_z))

    for t in timesteps:
        ellipse_id = 4
        E[ellipse_id, 3] = E[ellipse_id, 3] + np.sin(t/30*2*np.pi)*0.02

        volume_img = ct_shepp_logan((self.dim_x, self.dim_y), E=E)
        volume_img = torch.Tensor(np.flip(volume_img, axis=0).copy()) 
        volumes_list.append(volume_img)

        image_fat_layer = torch.clone(volume_img)
        image_fat_layer[image_fat_layer < 1] = 0 # Bright parts represent fat
        image_water_layer = volume_img - image_fat_layer  # Dim parts represent water

        xs, ys = torch.meshgrid(torch.arange(dim_x), torch.arange(dim_y), indexing='ij')
        z_offset = dim_z / 4 * make_gaussian(xs, ys, dim_x)
        self.training_volumes[t, xs, ys, (dim_z/5 + z_offset).long()] = image_water_layer
        self.training_volumes[t, xs, ys, (3*dim_z/4 + z_offset).long()] = image_fat_layer

    self.num = len(timesteps)
    all_angles = np.arange(start=np.pi, stop=np.pi + self.num*np.pi/5 , step=np.pi/5)
    self.training_angles = all_angles
    self.unique_angles = [float(f"{i:.2f}") for i in  all_angles[all_angles < 2*pi]]
    self.imag_volume = torch.zeros(size=(self.dim_x, self.dim_y, self.dim_z))

    def density_activation(density): return density
    aabb = torch.Tensor([[0, 0, 0], [self.dim_x, self.dim_y, self.dim_z]]).to(device)
    multiscale_res = [1, 2]
    grid_config = [{"grid_dimensions": 2, 
                "input_coordinate_dim": 4, 
                "output_coordinate_dim": 16,
                "resolution": [self.dim_x//multiscale_res[-1], self.dim_y//multiscale_res[-1], self.dim_z//multiscale_res[-1], self.num]}]
    
    self.real_kplane = KPlaneField(aabb=aabb, 
                                   grid_config = grid_config, 
                                   concat_features_across_scales = True, 
                                   multiscale_res = multiscale_res, 
                                   use_appearance_embedding = False, 
                                   appearance_embedding_dim = 0,
                                   density_activation = density_activation, 
                                   linear_decoder = True, 
                                   linear_decoder_layers = 1, 
                                   num_images = 0).to(device)
    
    self.imag_kplane = KPlaneField(aabb=aabb, 
                                   grid_config = grid_config, 
                                   concat_features_across_scales = True, 
                                   multiscale_res = multiscale_res, 
                                   use_appearance_embedding = False, 
                                   appearance_embedding_dim = 0,
                                   density_activation = density_activation, 
                                   linear_decoder = True, 
                                   linear_decoder_layers = 1, 
                                   num_images = 0).to(device)
    
    self.regularizers = self.get_regularizers(plane_tv_weight=plane_tv_weight, l1_time_planes=l1_time_planes, time_smoothness_weight=time_smoothness_weight)

    # Pre-generate masks for each unique training angle; Dictionary: key = training angle, value = mask, already on device; Lookup is based on the angle mod 2pi
    self.masks = [self.make_mask(theta*180/pi, N=200) for theta in self.unique_angles]
    self.angle_mask_dict = dict(zip(self.unique_angles, self.masks))

    ### Generate training and testing data => Truth values @ initialization
    self.training_data = []
    self.training_kspace = []
    for theta, timestep in zip(self.training_angles, timesteps):
      img_ren = self.render_img(theta, timestep, use_overhead = False, use_predicted = False)
      # Save copies of the raw projections for later use
      np.save(os.path.join(self.log_train, f'pre_blur_train_{timestep}.npy'), np.array(img_ren.cpu()))
      gt_kspace, gt_img = self.blur_img(img=img_ren, theta=theta)
      self.training_data.append(gt_img)
      self.training_kspace.append(gt_kspace)
      # Save copies of the masks for visualization
      mask = self.make_mask(theta, img_ren.shape[0])
      vis = mask.detach().cpu()
      np.save(os.path.join(self.log_train, f'mask_{timestep}.npy'), np.array(mask.cpu()))
      vis = np.asarray(vis*255).astype(np.uint8)
      imageio.imwrite(os.path.join(self.log_train, f'mask_{timestep}.png'), vis)
    self.training_data = torch.stack(self.training_data)
    self.training_kspace = torch.stack(self.training_kspace)
    
    self.testing_data = []
    for timestep in timesteps:
      self.testing_data.append(self.render_img(self.test_angle, timestep, use_overhead = True, use_predicted = False))

    timestamps = np.arange(self.num)
    self.timestamps = [((i - min(timestamps)) * (2.0 / (max(timestamps) - min(timestamps))) - 1.0) for i in timestamps]


  def make_mask(self, theta, N=200):
    im = Image.fromarray(np.uint8(np.zeros(self.training_volumes[0, :, :, 0].shape)))
    fig, ax = plt.subplots()
    _ = ax.imshow(im)
    
    width, height  = N, (N * np.sqrt((3-np.sqrt(5.)) / (5 + np.sqrt(5.))))
    x = (self.training_volumes[0, :, :, 0].shape[0])/2-width / 2
    y = (self.training_volumes[0, :, :, 0].shape[1])/2-height / 2
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
    ycrop = np.where(data[data.shape[0]//3,:,0]==0)[-1] # y min
    xcrop = np.where(data[:,data.shape[1]//3,0]==0)[-1] # x min
    cropped = data[xcrop[0]:xcrop[1], ycrop[0]:ycrop[1]].copy()

    # Make it black and white
    cropped[cropped < 200] = 0

    # Resize back to N by N
    im = Image.fromarray(cropped)
    im = im.resize(self.training_volumes[0, :, :, 0].shape)
    cropped = np.array(im)

    # Change data type
    cropped = np.array(cropped[:,:,0], dtype=np.float32) / 255
    mask = torch.from_numpy(cropped).to(device)

    assert mask.device == device
    return mask


  # Blur an image according to a PROPELLER blade k-space mask
  # Expects input to be complex, and returns a complex image
  def blur_img(self, img, theta):
    assert img.dtype == torch.complex64

    if (theta + 0.01)%(2*pi) < pi:
      theta = theta%(2*pi) + pi
    else:
      theta = theta%(2*pi)
    theta = float(f"{theta:.2f}")

    mask = self.angle_mask_dict[theta]
    img_FFT = torch.fft.fft2(img)
    img_DFT = torch.fft.fftshift(img_FFT)
    img_DFT_masked = img_DFT * mask
    inv_img_DFT_masked = torch.fft.ifftshift(img_DFT_masked)
    inv_img_DFT_masked = torch.fft.ifft2(inv_img_DFT_masked)
    return img_DFT_masked, inv_img_DFT_masked


  # Return the points of intersection of the ray based on measurement angle and the pixels
  def get_intersection_pts(self, pixel_x, pixel_y, measurement_angle, use_overhead):
    ### pixel_x and pixel_y are shape [W, H]
    if use_overhead:
      offset_angle = 0.1/180*pi
    else:
      offset_angle = 20/180*pi # This is a design parameter that is somewhat linked to the z (omega) resolution
    
    step_size = 0.5
    n_pts = (int)(self.dim_z * 2 / step_size) # upper bound
    x_step = step_size * np.cos(measurement_angle) * np.sin(offset_angle)
    y_step = step_size * np.sin(measurement_angle) * np.sin(offset_angle)
    z_step = step_size * np.cos(offset_angle)
    pts = torch.zeros(pixel_x.shape[0], pixel_x.shape[1], n_pts, 3, device=device)

    if (np.cos(measurement_angle) * np.sin(offset_angle) == 0):
      pts[:,:,:,0] = torch.zeros(n_pts, device=device)[None,None,0:n_pts] + torch.zeros(pixel_x.shape[0], pixel_x.shape[1], 1, device=device)
    else:
      pts[:,:,:,0] = torch.arange(start=0, end=0 + n_pts*x_step, step=x_step, device=device)[None,None,0:n_pts] + pixel_x[:,:,None]

    if (np.sin(measurement_angle) * np.sin(offset_angle) == 0):
      pts[:,:,:,1] = torch.zeros(n_pts, device=device)[None,None,0:n_pts] + torch.zeros(pixel_x.shape[0], pixel_x.shape[1], 1, device=device)
    else:
      pts[:,:,:,1] = torch.arange(start=0, end=0 + n_pts*y_step, step=y_step, device=device)[None,None,0:n_pts] + pixel_y[:,:,None]

    if (np.cos(offset_angle) == 0):
      pts[:,:,:,2] = torch.zeros(n_pts, device=device)[None,None,0:n_pts] + torch.zeros(pixel_x.shape[0], pixel_x.shape[1], 1, device=device)
    else:
      pts[:,:,:,2] = torch.arange(start=0, end=0 + n_pts*z_step, step=z_step, device=device)[None,None,0:n_pts] + torch.zeros(pixel_x.shape[0], pixel_x.shape[1], 1, device=device)

    return pts


  def render_pix(self, pixel_x, pixel_y, measurement_angle, timestep, use_overhead, use_predicted, use_real):
    ### pixel_x and pixel_y are 2D vectors with shape [W, H] the pixel indices we want to render in an image
    pixel_x = pixel_x.to(device=device)
    pixel_y = pixel_y.to(device=device)

    if use_predicted:
      if use_real:
        grid = self.real_kplane
      else:
        grid = self.imag_kplane
    else:
      if use_real:
        grid = self.training_volumes[timestep]
      else:
        grid = self.imag_volume
      
      return self.render_training(pixel_x, pixel_y, measurement_angle, grid, use_overhead)

    ### get intersection points (x, y, z)
    timestamp = self.timestamps[timestep]
    timestamps = timestamp * torch.ones(pixel_x.shape[0] * pixel_x.shape[1], device=device)
    pts = self.get_intersection_pts(pixel_x, pixel_y, measurement_angle, use_overhead)
    pts = pts.reshape(-1, pts.shape[-2], pts.shape[-1])
    densities, _ = grid.get_density(pts=pts, timestamps=timestamps)
    densities = densities.view(pixel_x.shape[0], pixel_x.shape[1], -1)

    pixel_value = torch.sum(densities, dim=-1)  # [W, H]

    del densities
    del timestamps
    del grid

    gc.collect()
    torch.cuda.empty_cache()

    return pixel_value
  
  
  def render_training(self, pixel_x, pixel_y, measurement_angle, grid, use_overhead):
    ### pixel_x and pixel_y are 2D vectors with shape [W, H] the pixel indices we want to render in an image
    ### get intersection points (x, y, z)
    pts = self.get_intersection_pts(pixel_x, pixel_y, measurement_angle, use_overhead) # xs, ys, zs, masks should each have shape [W, H, npts]
    xs = pts[:,:,:,0]
    ys = pts[:,:,:,1]
    zs = pts[:,:,:,2]
    n_pts = pts.shape[2]

    ### normalize, reshape and permute
    normalized_xs = (2*xs)/self.dim_x - 1
    normalized_ys = (2*ys)/self.dim_y - 1
    normalized_zs = (2*zs)/self.dim_z - 1

    xs = normalized_xs[None, None, ...]
    ys = normalized_ys[None, None, ...]
    zs = normalized_zs[None, None, ...]

    xs_perm = torch.permute(xs, dims=(4, 1, 2, 3, 0))
    ys_perm = torch.permute(ys, dims=(4, 1, 2, 3, 0))
    zs_perm = torch.permute(zs, dims=(4, 1, 2, 3, 0))

    ### points
    input_grid = grid[None, None, ...]
    input_grid = input_grid.repeat(n_pts, 1, 1, 1, 1).cuda()

    grid_input = torch.cat((zs_perm, ys_perm, xs_perm), -1).cuda()
    points_grid_samp = torch.nn.functional.grid_sample(input_grid, grid_input, mode='bilinear', padding_mode='zeros', align_corners=False).to(device=device)
    points = torch.permute(points_grid_samp, dims=(3, 4, 0, 1, 2)).squeeze()

    pixel_value = torch.sum(points, dim=-1)
    return pixel_value


  # Compute a projection of the volume at the measurement angle. Result is a 2D image.
  # Uses both real and imaginary parts of the model, and produces a complex image
  def render_img(self, measurement_angle, timestep, use_overhead = False, use_predicted = True):
    xs, ys = torch.meshgrid(torch.arange(self.training_volumes.shape[1]), torch.arange(self.training_volumes.shape[2]), indexing='ij')

    image_real = self.render_pix(xs, ys, measurement_angle, timestep, use_overhead, use_predicted, use_real=True)
    image_imag = self.render_pix(xs, ys, measurement_angle, timestep, use_overhead, use_predicted, use_real=False)
    # Combine into a single complex image
    image = torch.complex(image_real, image_imag).to(device)
    assert image.device == device
    assert image.dtype == torch.complex64
    return image


  def compute_loss(self, measurement_angle, timestep, gt_img, kspace_gt=None, use_overhead=False):
    img_predicted = self.render_img(measurement_angle, timestep, use_overhead) # rendered
    
    if use_overhead:
      log_dir = self.log_test
      mse_loss = torch.nn.functional.mse_loss(torch.abs(img_predicted), torch.abs(gt_img)) # compute the MSE between predicted image & ground truth = loss
    else:
      kspace_predicted, img_predicted = self.blur_img(img_predicted, measurement_angle) # rendered and blurred
      log_dir = self.log_train
      mse_loss = 0.00001 * torch.mean(torch.abs(torch.square(kspace_predicted - kspace_gt))) + 0.99999 * torch.mean(torch.abs(torch.square(img_predicted - gt_img)))

    ### Visualize/imwrite img_blurred and gt_img_blurred
    if (log_dir != None):
      vis = torch.abs(torch.cat((gt_img, img_predicted), dim=1)).detach()
      vis = vis - torch.min(vis)
      vis = np.asarray((vis/torch.max(vis) * 255).cpu()).astype(np.uint8)
      imageio.imwrite(os.path.join(log_dir, f'angle_{measurement_angle}_{timestep}.png'), vis)

    assert mse_loss.dtype == torch.float32
    reg_loss = 0
    for r in self.regularizers:
      reg_loss += r.regularize(self.real_kplane)

    del img_predicted

    return mse_loss, reg_loss


  def load_model(self, checkpoint_data):

    self.real_kplane.load_state_dict(checkpoint_data["real_model"], strict=False)
    self.imag_kplane.load_state_dict(checkpoint_data["imag_model"], strict=False)
    log.info("=> Loaded model state from checkpoint")

    self.optimizer.load_state_dict(checkpoint_data["optimizer"])
    log.info("=> Loaded optimizer state from checkpoint")


  def get_save_dict(self):
    return {
        "real_model": self.real_kplane.state_dict(),
        "imag_model": self.imag_kplane.state_dict(),
        "optimizer": self.optimizer.state_dict()
    }
  

  def save_planes(self, epoch):
    plane_names = ['xy', 'xz', 'xt', 'yz', 'yt', 'zt']

    for resolution_scale in range(len(self.real_kplane.multiscale_res_multipliers)):
      for plane, name in zip(self.real_kplane.grids[resolution_scale], plane_names):

        pred = torch.mean(plane.squeeze(), dim = 0)
        vis = pred.detach()
        vis = vis - torch.min(vis)
        vis = np.asarray((vis/torch.max(vis) * 255).cpu()).astype(np.uint8)
        os.makedirs(os.path.join(self.log_planes, f'epoch_{epoch}'), exist_ok=True)
        imageio.imwrite(os.path.join(os.path.join(self.log_planes, f'epoch_{epoch}'), f'plane_{name}_scale_{self.real_kplane.multiscale_res_multipliers[resolution_scale]}.png'), vis)


  def save_video(self, epoch, denoise=False):
    imgs = []
    mse = 0

    for timestep, test_img in zip(np.arange(self.num), self.testing_data): # loop through zip(timestamps, test images)
        rendered_img = self.render_img(measurement_angle = 0.1/180*pi, timestep = timestep, use_overhead = True) # measurement angle = offset (overhead) angle # render the current timestamp from overhead, without blurring
        
        rendered_and_gt_img = torch.cat((torch.abs(test_img), torch.abs(rendered_img)), dim = 1) # concatenate the rendered image with the test/gt image

        mse = mse + torch.nn.functional.mse_loss(torch.abs(rendered_img), torch.abs(test_img))
        imgs.append(rendered_and_gt_img)
        
    average_mse = mse/self.num # divide mse by number of frames, compute psnr and print
    psnr = -10*np.log10(np.array(average_mse.cpu()))

    write_video_to_file(file_name = os.path.join(self.log_test, f"epoch_{epoch}_psnr_{psnr}.mp4"), frames = imgs)
    print(f'test psnr is {psnr}')

    # nonlocal means denoising. This is slow, so only do it at the end.
    if denoise:
      print(f'doing nonlocal means denoising on the final video')
      patch_kw = dict(patch_size=5, patch_distance=10) 
      denoised_imgs = []
      maxval = torch.max(torch.stack(imgs).flatten())
      denoised_mse = 0
      for img in imgs:
        # Separate left and right halves for gt and ours
        gt = img[:,:img.shape[0]] / maxval
        ours = img[:,img.shape[0]:] / maxval
        # Denoise ours
        denoised = torch.from_numpy(denoise_nl_means(np.array((ours*255).cpu()).astype(np.uint8), h=0.03, fast_mode=False, **patch_kw)).to(device)
        # Compute denoised PSNR
        denoised_mse = denoised_mse + torch.nn.functional.mse_loss(denoised, gt)
        denoised_imgs.append(torch.cat((gt, denoised), dim=1))
      denoised_psnr = -10*np.log10(np.array((denoised_mse/self.num).cpu()))
      write_video_to_file(file_name = os.path.join(self.log_test, f"epoch_{epoch}_psnr_{denoised_psnr}_denoised.mp4"), frames = denoised_imgs) 
      print(f'denoised test psnr is {denoised_psnr}') 


  def get_regularizers(self, plane_tv_weight, l1_time_planes, time_smoothness_weight):
     return [
          PlaneTV(plane_tv_weight, what='field'),
          L1TimePlanes(l1_time_planes, what='field'),
          TimeSmoothness(time_smoothness_weight, what='field')
      ]


  # Function to train and run optimization loop. Includes all functions as components to overall testing of prediction
  def train(self, num_epochs, lr):
    self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
    pb = tqdm(total=num_epochs)

    for epoch in range(num_epochs): ### loop over the dataset multiple times
        # Generate a random shuffle for this epoch
        permutation = np.random.permutation(self.num)
        count = 0
        train_loss_total = 0

        for index in tqdm(permutation):
            measurement_angle = self.training_angles[index]
            training_img = self.training_data[index,:,:]
            train_kspace = self.training_kspace[index,:,:]

            self.optimizer.zero_grad() ### zero the parameter gradients
            
            ### compute the loss based on model output and the ground truth
            mse, tv = self.compute_loss(measurement_angle, timestep=index, gt_img=training_img, kspace_gt=train_kspace, use_overhead=False)
            
            loss = mse + tv
            train_loss_total += float(loss)
            count += 1
            loss.backward() ### backpropagate the loss
            self.optimizer.step() ### adjust parameters based on the calculated gradients


        avg_train_loss = train_loss_total/count

        ### Compute and print the average accuracy for this epoch when tested over all test images
        pb.set_postfix_str(f'Epoch {epoch+1}: PSNR={-10*np.log10(avg_train_loss):.4f}', refresh=False)
        pb.update(1)

        if epoch % 1 == 0:
          self.save_planes(epoch)
          with torch.no_grad():
            self.save_video(epoch, denoise=(epoch+1==num_epochs))

    pb.close()


if __name__ == "__main__":
  xdim = 256
  ydim = 256
  zdim = 20
  frame = 1
  tv_lambda = 0.05
  l1_lambda = 0.005
  ts_lambda = 0.3
  lr = 0.004
  model = PropMRIOffResonanceCorrection(xdim, ydim, zdim, 
                                        expname=f'dynamic_shepp_xy{xdim}_z{zdim}_tv{tv_lambda}_l1{l1_lambda}_ts{ts_lambda}_lr{lr}', 
                                        plane_tv_weight = tv_lambda, l1_time_planes = l1_lambda, time_smoothness_weight = ts_lambda)
  model.train(50, lr=lr)
