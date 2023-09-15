class Path(object):
    @staticmethod
    def get_path_of(name):
        if name == "kitti":
            return '/zhouzm/Datasets/kitti'
        elif name == "kitti_stereo2015":
            return '/zhouzm/Datasets/kitti_2015'
        elif name == 'swin':
            return '/zhouzm/Download/swin_tiny_patch4_window7_224.pth'
        else:
            raise NotImplementedError