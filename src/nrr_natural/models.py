"""Control the newly initialized blocks when comparing retained versus fresh weights."""
import copy
from nrr.models import make_candidate

def candidate(teacher,kind,seed):
    model=make_candidate(teacher,kind,seed)
    if kind=='small_student':
        matched=make_candidate(teacher,'regionwise',seed)
        model.r0=copy.deepcopy(matched.r0)
        model.r1=copy.deepcopy(matched.r1)
    return model.eval()
