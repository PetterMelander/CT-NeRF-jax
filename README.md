# CT-NeRF

This is a private project started by me during the fall of 2024. It aims to use a modified version of the the work presented in [NeRF: Representing Scenes as
Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934) to construct a CT image from a series of X-ray images from different angles. 

Modifications include:
- Using the trained model to construct a CT image by sampling the model in a 3D grid rather than along rays. 
- Training the model to predict only [linear attenuation coefficients](https://en.wikipedia.org/wiki/Attenuation_coefficient) (what the authors of the orinial NeRF paper call density, $\sigma$). Colour is not needed since a CT image is just a rescaled mapping of linear attenuation coefficients. The output of a ray $\mathbf{r}$ is thus its transmittance $T(\mathbf{r})$ rather than its colour. 
- Removing the viewing angle dependent part of the original NeRF model. The angle dependent part is not needed since colours are not being predicted, and X-ray absorption does not depend on incidence angle of the ray.
- Simplifying the calculation of the output pixel values to a modified version of the [Beer-Lambert Law](https://en.wikipedia.org/wiki/Beer%E2%80%93Lambert_law): $T\left(\mathbf{r}\right)=\frac{\ln\left(\exp\left[\sum_{i=1}^{N}{\sigma_i\delta_i}\right]+k\right)}{s}$. The logarithm and scaling parameters $s$ and $k$ were added to compensate for the fact the the X-ray image intensities had been scaled by $T_{new} = \frac{\ln\left(T_{old}+k\right)}{s}$ to enhance the contrast. The transmittance in a raw digital X-ray image is close to 0 throughout the entire body, with only minute differences visible. Scaling and taking the logarithm can greatly enhance the contrast, improving the model's ability to learn. 
- Implementing a fine sampling scheme focused on edges in the output of the coarse model. A PDF of $\hat{w}_i$ defined by $w_i =\frac{\left|\sigma_{i+1}-\sigma_{i}\right|}{\delta_i}$, $\hat{w}_i=w_i/\sum_{j=1}^{N_c}{w_j}$ was used to create this effect, which helps the fine model focus on learning where edges between tissues are located. This helps give clearly defined organ boundaries. 

## Implementation details

The NeRF model and ray tracing was implemented in PyTorch. To make the project more challenging, everything was written by hand, no code was taken from existing NeRF implementations. Care was taken to ensure the code runs efficiently. The NeRF model was trained to construct its representation of the CT image in a cube of size $2\times 2\times 2$ with its centre at the origin. 

### Ray tracing

A ray in three dimensions can be parameterized by a single scalar $t$ as $\mathbf{p}+t\cdot\mathbf{v}$, with $\mathbf{p}$ being a point on the ray and $\mathbf{v}$ being the heading vector of the ray. For an X-ray image of size $(w_{im}, h_{im})$ taken from an angle $0$, that is head on, the starting position $\mathbf{p}(y, z, 0)$ and heading vector $\mathbf{v}(0)$ of the ray associated with pixel $(y,z)$ are 

$$\mathbf{p}(y, z, 0)=\begin{bmatrix}1 & \frac{2y}{w_{im}}-1 & \frac{2z}{h_{im}}-1\end{bmatrix}^T \\ 

\mathbf{v}(0)=\begin{bmatrix}-1 & 0 & 0 \end{bmatrix}^T$$ 

That is, the ray starts at $x=1$ and heads in the negative x direction. This works because unlike in a normal camera, in an X-ray camera the light rays creating the image are more or less parallel and orthogonal to the image plane. Since the start position is at $x=1$, $\mathbf{v}(0)$ is a unit vector, and the model space is a cube with a side of $2$, it holds that $t \in [0,2]$. 

For pixel $(y,z)$ in an X-ray image taken from angle $\theta$, the starting point $\mathbf{p}(y,z,\theta)$ and heading vector $\mathbf{v}(\theta)$ can be calculated by rotating $\mathbf{p}(y, z, 0)$ and $\mathbf{v}(0)$ by $\theta$ around the z-axis: 

$$\mathbf{R}_z(\theta)=\begin{bmatrix} \cos{\theta} & -\sin{\theta} & 0 \\ \sin{\theta} & \cos{\theta} & 0 \\ 0 & 0 & 1\end{bmatrix} \\ 

\mathbf{p}(y, z, \theta)=\mathbf{R}_z(\theta)\mathbf{p}(y,z,0)\\ 

\mathbf{v}(\theta)=\mathbf{R}_z(\theta)\mathbf{v}(0)=\begin{bmatrix} \cos{\theta} & -\sin{\theta} & 0\end{bmatrix}^T$$

Thus, the ray associated with pixel $(y,z)$ of an X-ray image taken from angle $\theta$ can be parameterized as $\mathbf{p}(y,z,\theta)+t\cdot\mathbf{v}(\theta)$ with $t\in[0,2]$. Sampling points along the ray is then as simple as sampling different values of $t$.

This sampling scheme does mean that the edges of the image will not be reached by rays from all angles, since the size of the cube is greater than $2$ along some directions. Also, some rays will reach outside the cube. This should not be a problem however, as the middle of the image, where the subject is located, is hit by rays from every angle. An alternative sampling scheme where samples of $t$ are only drawn from the central cylinder of the cube, $x^2+y^2=1$, was also implemented and tested. For a starting position $\begin{bmatrix} a & b & c\end{bmatrix}^T$ and heading vector $\begin{bmatrix} v_x & v_y & 0\end{bmatrix}^T$, this corresponds to $t\in\left[-(av_x+bv_y)\pm\sqrt{(av_x+bv_y)^2-(a^2+b^2-1)}\right]$. 

### Image I/O

Reading and writing CT and X-ray images was done using [SimpleITK](https://simpleitk.org/). Because of this, the X-ray images can be supplied in any file format that can be read by SimpleITK, and the CT image can be outputted in any file format SimpleITK can write. The CT image for creating simulated X-ray images (see below) can also be in any file format supported by SimpleITK. 



Skriv om json-metadata.


Skriv om att enheter och beer-lambert gör att modellen faktiskt outputtar riktigt absorptionskoefficienter. 

## X-ray creation

Since I could not find any datasets consisting of whole body X-ray images from multiple angles, with exact angle annotations (why would such a dataset even exist?), I wrote code for creating simulated X-ray images from a CT image. This involves rescaling from [Hounsfield Units](https://en.wikipedia.org/wiki/Hounsfield_scale) to linear attenuation coefficients and applying a discretized version of the Beer-Lambert law along the desired angle. In practice, applying the Beer-Lambert law along an angle $\theta$ involves rotating the CT image $\theta$ degrees and applying the Beer-Lambert law along the x-axis.

## Results

Ha med bilder!

## Ablation study