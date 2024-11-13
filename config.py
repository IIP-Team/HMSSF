class DefaultConfigs(object):
    # model
    gpus = '0'
    model_type = 'LSFAT_Shunted_SpaSpe_NAE'     # LSFAT or LSFAT_WoCToken or LSFAT_Dilate or LSFAT_Shunted or transformer or 3DCNN
    # training parameters
    train_epoch = 100
    BATCH_SIZE_TRAIN = 64
    test_epoch = 10
    # source data information
    data = 'PaviaU'   #PaviaU-9 / Indian-16
    num_classes = 9
    # patch size
    patch_size = 15
    # The proportion of test samples
    test_ratio = 0.95
    pca_components = 30

    # 3DConv parameters
    dim_3Dout = 5
    dim_3DKernel1 = 3
    dim_3DKernel23 = 3
    #spatial branch
    spa_downks = [4, 1]  # max[4, 1]
    dim1 = 60
    dim2 = 60
    dim3 = 120
    num_heads = 3
    K_SPA = [1, 2, 3]  #[1, 2, 3, 4, 5] #[1, 2, 3]  [1, 3, 6]
    dim_patch = patch_size - dim_3DKernel23 + 1
    dim_linear = pca_components - dim_3DKernel1 + 1
    # spectral branch
    spe_downks = [4, 2]
    dim1_SPE = dim1
    dim2_SPE = dim2
    dim3_SPE = dim3
    K_SPE = [1, 5, 10]  #[1, 5, 10, 15, 20] #[1, 2, 3]
    # common parameters
    dim_classes = 240

    # paths information
    checkpoint_path = './' + "checkpoint/" + data + '/' + model_type + '/' + 'TrainEpoch' + str(train_epoch) + '_TestEpoch' + str(test_epoch) + '_Batch' + str(BATCH_SIZE_TRAIN) + '/' \
                      + 'PatchSize' + str(patch_size) + '/' + '3DConv' + str([dim_3Dout,  dim_3DKernel1,  dim_3DKernel23,  dim_3DKernel23]) + '/' \
                      + 'Dim' + str([dim1, dim2, dim3]) + '_DimClasses' + str(dim_classes) + '/' + 'NumHeads' + str(num_heads) + '_KSpa' + str(K_SPA) + '_KSpe' + str(K_SPE) + '/' \
                      + 'SpaDownKS' + str(spa_downks)+ '_SpeDownKS' + str(spe_downks) + '/' + 'Test_ratio' + str(test_ratio) + '/'
    logs = checkpoint_path

config = DefaultConfigs()
