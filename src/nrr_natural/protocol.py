"""A versioned, content-addressed protocol fixed before training or test access."""
from dataclasses import dataclass, asdict
import hashlib
import json
from nrr.budget import positive_int
from nrr.experiment import METHODS

@dataclass(frozen=True)
class StudyProtocol:
    seeds: tuple[int, ...] = (17,29,43)
    budget_steps: tuple[int, ...] = (80,320)
    teacher_steps: int = 3000
    width: int = 16
    train_per_class: int = 1000
    val_per_class: int = 200
    split_seed: int = 20260917
    test_limit: int | None = None
    threads: int = 2
    batch_size: int = 32
    calibration_images: int = 128
    schema: str = 'nrr.natural-protocol.v1'

    def __post_init__(self):
        for name in ('teacher_steps','width','train_per_class','val_per_class','threads','batch_size','calibration_images'):
            positive_int(getattr(self,name),name)
        if self.width<4 or self.width%2:raise ValueError('width must be even and >=4')
        for name in ('seeds','budget_steps'):
            values=getattr(self,name)
            if not isinstance(values,tuple) or not values or len(set(values))!=len(values):raise ValueError(f'{name} must be a nonempty unique tuple')
            for v in values:positive_int(v,name)
        if any(s>=2**32-200 for s in self.seeds):raise ValueError('seed out of range')
        if tuple(sorted(self.budget_steps))!=self.budget_steps:raise ValueError('budgets must be increasing')
        if not isinstance(self.split_seed,int) or isinstance(self.split_seed,bool) or not 0<=self.split_seed<2**32:raise ValueError('Invalid split seed')
        if self.test_limit is not None:positive_int(self.test_limit,'test_limit')
        if self.batch_size!=32:raise ValueError('v1 fixed training batch size is 32')
        if self.schema!='nrr.natural-protocol.v1':raise ValueError('Unknown protocol schema')

    @property
    def expected_models(self):return len(self.seeds)*(1+len(METHODS)*len(self.budget_steps))

    def to_dict(self):
        return {**asdict(self),'seeds':list(self.seeds),'budget_steps':list(self.budget_steps),
                'methods':list(METHODS),'expected_models':self.expected_models,
                'initialization':'matched replacement blocks; student stem/head freshly initialized',
                'loss':'0.5 CE + 0.5 KL(T=2,T^2) + 0.1 final-region feature MSE',
                'learning_rate':0.04,'augmentation':'none','optimizer':'SGD momentum=.9 weight_decay=1e-4',
                'budget_unit':'(3 * student Conv/Linear MACs + teacher Conv/Linear MACs) * images',
                'selection':'validation only; initial plus five budget-fraction checkpoints; earliest ties',
                'test_policy':'freeze every selected checkpoint before first test evaluation',
                'compiler':'ethos-u-vela==5.1.0, ethos-u55-256; compiler estimates are not hardware measurements',
                'inference':'pre-Vela TFLite BUILTIN_REF on CPU; no NPU inference',
                'interpretation':'exploratory fixed architecture/subset; no automated winner or novelty claim'}

    @property
    def protocol_id(self):
        return hashlib.sha256(json.dumps(self.to_dict(),sort_keys=True,separators=(',',':')).encode()).hexdigest()
