__version__ = '0.4.2'
git_version = 'b2946be43ddc7b51ce186ae09d87188bbab05dcf'
from torchvision import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
