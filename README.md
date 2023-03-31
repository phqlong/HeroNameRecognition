Hero Name Recognition
==============================

League of Legends Hero Name Recognition Project

Project Organization
------------

```
├── HeroDetection                    <- Main folder for Pytorch Lightning Code for FineTuing Backbone
│   ├── src                          <- Source code
│       ├── datamodules              <- Lightning datamodules
│       ├── models                   <- Lightning models
│       ├── utils                    <- Utility scripts
│       │
│       ├── eval.py                  <- Run evaluation
│       └── train.py                 <- Run training
│
├── model                            <- Model checkpoint and ANN model
├── test_data                        <- all provided data and crawled images
├── crawling.py                      <- Scripts for crawling hero images
├── test.ipynp                       <- Notebook for demo
│       
├── test.py                          <- Main code for running test result
│       
├── utils.py                         <- Some utils served for test.py
├── requirements.txt                 <- All needed libs
```

==============================
How to run
------------
For running with served model backbone and ANN indices, run:

```
python test.py
```

For finetunning backbone again:

```
cd HeroDetection
python src/train.py
```
