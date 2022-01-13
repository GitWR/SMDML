# SMDML-v1.0

(1) This is a matlab implementation of the proposed SMDML. 

This work carries out a preliminary exploration of the combination of deep Riemannian neural network and Riemannian metric learning to find a possible way to simultaneously conquer the degradation of structural information during multi-stage feature transformation and reduce the impact of the large intra-class variations and inter-class ambiguity of representations on the model capacity. The experimental results on the four benchmarking datasets confirm its validity.

(2) This code is constructed on the basis of SPDNet [1]. We are very grateful for the very efficient and reliable source code provided by the authors.
    
    [1] Huang. Z and Van. G. L. A riemannian network for spd matrix learning. In AAAI, 2017, pp. 2036-2042.

(3) Here, we use the FPHA dataset [2] as an example. 

    [2] Garcia-Hernando. G, Yuan. S, Baek. S and Kim. T. K. First-person hand action benchmark with RGB-D videos and 3D hand pose annotations. 
        In CVPR, 2018, pp. 409-419.

(4) To rerun this code, the following steps are required: 

    1) MATLAB R2018 software or higher version; 
    
    2) Deep learning toolbox (currently, this toolbox is not required, but could be used for acceleration later);
    
    3) Building the folders './data/afew' in the current path;
    
    4) Placing the FPHA folder and the SPD_info.mat file into the path of './data/afew';
    
    3) run spdnet_afew.m

(5) For any questions, please do not hesitate to contact me at: cs_wr@jiangnan.edu.cn
