# GLIN
GLIN: Global and local information integrated network for remaining useful life prediction

Under a window-manner end-to-end paradigm, data-driven RUL prediction methods suffer from unsatisfying generalization ability and low interpretability, as the consequence of neglecting diverse modes among the entire degradation processes of different entities.

We propose GLIN [[paper]](https://doi.org/10.1016/j.engappai.2023.106956) to integrate global and local information. 

![image](https://github.com/zacchen14/RUL_GLIN/blob/main/pic/GLIN.png)

## Get Started
1. Install the required packages.
   
   ``$ pip install -r requirements.txt``

2. Prepare data. You can obtain the C-MAPSS Aircraft Engine Simulator Data from [NASA's Open Data Portal](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository). 

3. Train the model.

   `$ python main.py --train_dataset FD001 --test_dataset FD001 --epochs 100`
