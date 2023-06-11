from lib.algos.gans import RCGAN, RCWGAN, TimeGAN, CWGAN
from lib.algos.gmmn import GMMN
from lib.algos.sigcwgan import SigCWGAN
from lib.algos.asganadv import AECGAN


ALGOS = dict(
    SigCWGAN=SigCWGAN, 
    AECGAN=AECGAN,
    TimeGAN=TimeGAN, 
    RCGAN=RCGAN, 
    GMMN=GMMN, 
    RCWGAN=RCWGAN, 
    CWGAN=CWGAN)
