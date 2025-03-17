from model import Model
from training import Training


model = Model()

trainer = Training(model)

losses = trainer.train(batch_size=64, epochs=5)
