import numpy as np
import torch
import torch.nn.functional as F
import cv2
from . import networks

def process_uv(uv_coords, uv_h=224, uv_w=224):
    uv_coords[:, 0] = uv_coords[:, 0] * (uv_w - 1)
    uv_coords[:, 1] = uv_coords[:, 1] * (uv_h - 1)
    uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1))))
    return uv_coords

def bilinear_interpolate(img, x, y):
    x0 = torch.floor(x).long()
    x1 = x0 + 1
    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, img.shape[1] - 1)
    x1 = torch.clamp(x1, 0, img.shape[1] - 1)
    y0 = torch.clamp(y0, 0, img.shape[0] - 1)
    y1 = torch.clamp(y1, 0, img.shape[0] - 1)

    i_a = img[y0, x0]
    i_b = img[y1, x0]
    i_c = img[y0, x1]
    i_d = img[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return wa.unsqueeze(-1) * i_a + wb.unsqueeze(-1) * i_b + wc.unsqueeze(-1) * i_c + wd.unsqueeze(-1) * i_d

class face_model:
    def __init__(self):
        # Hardcoded settings
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.backbone = 'resnet50'  # Options: 'resnet50' or 'mbnetv3'
        self.extract_texture = True  # Hardcode extractTex=True
        self.use_landmarks_68 = True  # Hardcode ldm68=True
        self.use_landmarks_106 = True  # Hardcode ldm106=True
        self.use_landmarks_106_2d = True  # Hardcode ldm106_2d=True
        self.use_landmarks_134 = True  # Hardcode ldm134=True
        self.use_seg = True  # Hardcode seg=True
        self.use_seg_visible = True  # Hardcode seg_visible=True

        # Load face model data
        model = np.load("./assets/face_model.npy", allow_pickle=True).item()
        self.u = torch.tensor(model['u'], requires_grad=False, dtype=torch.float32, device=self.device)
        self.id = torch.tensor(model['id'], requires_grad=False, dtype=torch.float32, device=self.device)
        self.exp = torch.tensor(model['exp'], requires_grad=False, dtype=torch.float32, device=self.device)
        self.u_alb = torch.tensor(model['u_alb'], requires_grad=False, dtype=torch.float32, device=self.device)
        self.alb = torch.tensor(model['alb'], requires_grad=False, dtype=torch.float32, device=self.device)
        self.point_buf = torch.tensor(model['point_buf'], requires_grad=False, dtype=torch.int64, device=self.device)
        self.tri = torch.tensor(model['tri'], requires_grad=False, dtype=torch.int64, device=self.device)
        self.uv_coords = torch.tensor(model['uv_coords'], requires_grad=False, dtype=torch.float32, device=self.device)

        # Texture extraction setup
        if self.extract_texture:
            uv_coords_numpy = process_uv(model['uv_coords'].copy(), 1024, 1024)
            self.uv_coords_torch = (torch.tensor(uv_coords_numpy, requires_grad=False, dtype=torch.float32, device=self.device) / 1023 - 0.5) * 2
            if self.device == 'cpu':
                from util.cpu_renderer import MeshRenderer_UV_cpu
                self.uv_renderer = MeshRenderer_UV_cpu(rasterize_size=1024)
                self.uv_coords_torch += 1e-6  # Slight perturbation for CPU renderer
            else:
                from util.nv_diffrast import MeshRenderer_UV
                self.uv_renderer = MeshRenderer_UV(rasterize_size=1024)
            self.uv_coords_numpy = uv_coords_numpy.copy()
            self.uv_coords_numpy[:, 1] = 1024 - self.uv_coords_numpy[:, 1] - 1

        # Landmark indices
        if self.use_landmarks_68:
            self.ldm68 = torch.tensor(model['ldm68'], requires_grad=False, dtype=torch.int64, device=self.device)
        if self.use_landmarks_106 or self.use_landmarks_106_2d:
            self.ldm106 = torch.tensor(model['ldm106'], requires_grad=False, dtype=torch.int64, device=self.device)
        if self.use_landmarks_134:
            self.ldm134 = torch.tensor(model['ldm134'], requires_grad=False, dtype=torch.int64, device=self.device)

        # Segmentation setup
        if self.use_seg_visible:
            self.annotation = model['annotation']
        if self.use_seg:
            self.annotation_tri = [torch.tensor(i, requires_grad=False, dtype=torch.int64, device=self.device) for i in model['annotation_tri']]

        # 2D landmarks setup
        if self.use_landmarks_106_2d:
            self.parallel = model['parallel']
            self.v_parallel = -torch.ones(35709, device=self.device, dtype=torch.int64)
            for i, p in enumerate(self.parallel):
                self.v_parallel[p] = i

        # Renderer setup
        self.persc_proj = torch.tensor([1015.0, 0, 112.0, 0, 1015.0, 112.0, 0, 0, 1], dtype=torch.float32, device=self.device).reshape(3, 3).T
        self.camera_distance = 10.0
        if self.device == 'cpu':
            from util.cpu_renderer import MeshRenderer_cpu
            self.renderer = MeshRenderer_cpu(rasterize_fov=2 * np.arctan(112. / 1015) * 180 / np.pi, znear=5., zfar=15., rasterize_size=224)
        else:
            from util.nv_diffrast import MeshRenderer
            self.renderer = MeshRenderer(rasterize_fov=2 * np.arctan(112. / 1015) * 180 / np.pi, znear=5., zfar=15., rasterize_size=224)

        # Load network
        if self.backbone == 'resnet50':
            self.net_recon = networks.define_net_recon(net_recon='resnet50', use_last_fc=False, init_path=None)
            weights_path = "assets/net_recon.pth"  # Adjust to your ResNet-50 weights path
        else:  # Default to MobileNet-V3
            self.net_recon = networks.define_net_recon_mobilenetv3(net_recon='recon_mobilenetv3_large', use_last_fc=False, init_path=None)
            weights_path = "assets/net_recon_mbnet.pth"  # Provided in repo
        self.net_recon.load_state_dict(torch.load(weights_path, map_location='cpu')['net_recon'])
        self.net_recon = self.net_recon.to(self.device).eval()

        self.input_img = None

    def compute_shape(self, alpha_id, alpha_exp):
        batch_size = alpha_id.shape[0]
        face_shape = torch.einsum('ij,aj->ai', self.id, alpha_id) + torch.einsum('ij,aj->ai', self.exp, alpha_exp) + self.u.reshape(1, -1)
        return face_shape.reshape(batch_size, -1, 3)

    def compute_albedo(self, alpha_alb):
        batch_size = alpha_alb.shape[0]
        face_albedo = torch.einsum('ij,aj->ai', self.alb, alpha_alb) + self.u_alb.reshape(1, -1)
        return (face_albedo / 255.).reshape(batch_size, -1, 3)

    def compute_norm(self, face_shape):
        v1 = face_shape[:, self.tri[:, 0]]
        v2 = face_shape[:, self.tri[:, 1]]
        v3 = face_shape[:, self.tri[:, 2]]
        e1 = v1 - v2
        e2 = v2 - v3
        face_norm = torch.cross(e1, e2, dim=-1)
        face_norm = F.normalize(face_norm, dim=-1)
        face_norm = torch.cat([face_norm, torch.zeros(face_norm.shape[0], 1, 3, device=self.device)], dim=1)
        vertex_norm = torch.sum(face_norm[:, self.point_buf], dim=2)
        return F.normalize(vertex_norm, dim=-1)

    def compute_rotation(self, angles):
        batch_size = angles.shape[0]
        ones = torch.ones([batch_size, 1], device=self.device)
        zeros = torch.zeros([batch_size, 1], device=self.device)
        x, y, z = angles[:, :1], angles[:, 1:2], angles[:, 2:]
        rot_x = torch.cat([ones, zeros, zeros, zeros, torch.cos(x), -torch.sin(x), zeros, torch.sin(x), torch.cos(x)], dim=1).reshape(batch_size, 3, 3)
        rot_y = torch.cat([torch.cos(y), zeros, torch.sin(y), zeros, ones, zeros, -torch.sin(y), zeros, torch.cos(y)], dim=1).reshape(batch_size, 3, 3)
        rot_z = torch.cat([torch.cos(z), -torch.sin(z), zeros, torch.sin(z), torch.cos(z), zeros, zeros, zeros, ones], dim=1).reshape(batch_size, 3, 3)
        rot = rot_z @ rot_y @ rot_x
        return rot.permute(0, 2, 1)

    def to_camera(self, face_shape):
        face_shape[..., -1] = self.camera_distance - face_shape[..., -1]
        return face_shape

    def to_image(self, face_shape):
        face_proj = face_shape @ self.persc_proj
        return face_proj[..., :2] / face_proj[..., 2:]

    def transform(self, face_shape, rot, trans):
        return face_shape @ rot + trans.unsqueeze(1)

    def split_alpha(self, alpha):
        return {
            'id': alpha[:, :80],
            'exp': alpha[:, 80:144],
            'alb': alpha[:, 144:224],
            'angle': alpha[:, 224:227],
            'sh': alpha[:, 227:254],
            'trans': alpha[:, 254:]
        }

    def get_landmarks_68(self, v2d):
        return v2d[:, self.ldm68]

    def get_landmarks_106(self, v2d):
        return v2d[:, self.ldm106]

    def get_landmarks_134(self, v2d):
        return v2d[:, self.ldm134]

    def get_landmarks_106_2d(self, v2d, face_shape, alpha_dict):
        temp_angle = alpha_dict['angle'].clone()
        temp_angle[:, 2] = 0
        rotation_without_roll = self.compute_rotation(temp_angle)
        v2d_without_roll = self.to_image(self.to_camera(self.transform(face_shape, rotation_without_roll, alpha_dict['trans'])))
        ldm106_dynamic = self.ldm106.clone()
        for i in range(16):
            temp = v2d_without_roll.clone()[:, :, 0]
            temp[:, self.v_parallel != i] = 1e5
            ldm106_dynamic[i] = torch.argmin(temp)
        for i in range(17, 33):
            temp = v2d_without_roll.clone()[:, :, 0]
            temp[:, self.v_parallel != i] = -1e5
            ldm106_dynamic[i] = torch.argmax(temp)
        return v2d[:, ldm106_dynamic]

    def forward(self):
        alpha = self.net_recon(self.input_img)
        alpha_dict = self.split_alpha(alpha)
        face_shape = self.compute_shape(alpha_dict['id'], alpha_dict['exp'])
        rotation = self.compute_rotation(alpha_dict['angle'])
        face_shape_transformed = self.transform(face_shape, rotation, alpha_dict['trans'])
        v3d = self.to_camera(face_shape_transformed)
        v2d = self.to_image(v3d)

        result_dict = {
            'v3d': v3d.detach().cpu().numpy(),
            'v2d': v2d.detach().cpu().numpy(),
            'tri': self.tri.detach().cpu().numpy(),
            'uv_coords': self.uv_coords.detach().cpu().numpy(),
        }

        if self.use_landmarks_68:
            result_dict['ldm68'] = self.get_landmarks_68(v2d).detach().cpu().numpy()
        if self.use_landmarks_106:
            result_dict['ldm106'] = self.get_landmarks_106(v2d).detach().cpu().numpy()
        if self.use_landmarks_106_2d:
            result_dict['ldm106_2d'] = self.get_landmarks_106_2d(v2d, face_shape, alpha_dict).detach().cpu().numpy()
        if self.use_landmarks_134:
            result_dict['ldm134'] = self.get_landmarks_134(v2d).detach().cpu().numpy()

        return result_dict
