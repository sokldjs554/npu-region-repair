"""MAC-equivalent budget. This excludes normalization and memory traffic."""
from dataclasses import dataclass


def positive_int(value: int, name: str) -> None:
    if not isinstance(value,int) or isinstance(value,bool) or value<=0:
        raise ValueError(f'{name} must be a positive integer')


@dataclass
class Budget:
    cap: int
    spent: int=0

    def __post_init__(self):
        positive_int(self.cap,'cap')
        if self.spent!=0: raise ValueError('A new budget must start at zero')

    @property
    def remaining(self)->int: return self.cap-self.spent

    def can_charge(self, units: int)->bool:
        positive_int(units,'units')
        return self.spent+units<=self.cap

    def charge(self, units: int)->None:
        if not self.can_charge(units): raise ValueError('Training budget exceeded')
        self.spent+=units


def batch_charge(student_macs: int, teacher_macs: int, images: int)->int:
    for name,v in [('student_macs',student_macs),('teacher_macs',teacher_macs),('images',images)]: positive_int(v,name)
    return (3*student_macs+teacher_macs)*images
