from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore, EfficientAd
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = Patchcore(
    precision="float16"
)
#model = model.half()

datamodule = MVTecAD(
    root="./datasets/MVTecAD",
    category="bottle",
    train_batch_size=1,
    eval_batch_size=1,  
    num_workers=0
)

engine = Engine(
    accelerator="auto", 
    devices=1,  
    max_epochs=10,
)

engine.fit(
    model=model,
    datamodule=datamodule,
)

engine.test(model=model, datamodule=datamodule)


#3.4 GB GPU