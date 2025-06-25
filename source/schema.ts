export const typeDefs = `#graphql
    # ---------- Layer Config ----------
    interface LayerConfig {
        id: ID! 
        type: String! 
        name: String 
    }
    type LinearLayerConfig implements LayerConfig {
        id: ID! 
        type: String!   
        name: String    
        in_features: Int!  
        out_features: Int! 
    }
    type Conv2dLayerConfig implements LayerConfig {
        id: ID!
        type: String!
        name: String
        in_channels: Int!
        out_channels: Int!
        kernel_size: [Int!]!
        stride: [Int]
        padding: [Int]
        dilation: [Int]
        groups: [Int]
        bias: Boolean
        padding_mode: String
    }
    type Conv1dLayerConfig implements LayerConfig {
        id: ID!
        type: String!
        name: String
        in_channels: Int!
        out_channels: Int!
        kernel_size: [Int!]!
        stride: [Int]
        padding: [Int]
        dilation: [Int]
        groups: [Int]
        bias: Boolean
        padding_mode: String
    }
    type MaxPool2dLayerConfig implements LayerConfig {
        id: ID!
        type: String!
        name: String
        kernel_size: [Int!]!
        stride: [Int]
        padding: [Int]
        dilation: [Int]
        return_indices: Boolean
        ceil_mode: Boolean
    }
    type MaxPool1dLayerConfig implements LayerConfig {
        id: ID!
        type: String!
        name: String
        kernel_size: [Int!]!
        stride: [Int]
        padding: [Int]
        dilation: [Int]
        return_indices: Boolean
        ceil_mode: Boolean
    }
    type AvgPool2dLayerConfig implements LayerConfig {
        id: ID!
        type: String!
        name: String
        kernel_size: [Int!]!
        stride: [Int]
        padding: [Int]
        count_include_pad: Boolean
        divisor_override: Int
        ceil_mode: Boolean
    }
    type AvgPool1dLayerConfig implements LayerConfig {
        id: ID!
        type: String!
        name: String
        kernel_size: [Int!]!
        stride: [Int]
        padding: [Int]
        count_include_pad: Boolean
        ceil_mode: Boolean
    }
    type BatchNorm2dLayerConfig implements LayerConfig {
        id: ID!
        type: String!
        name: String
        num_features: Int!
        eps: Float
        momentum: Float
        affine: Boolean
        track_running_status: Boolean
    }
    type BatchNorm1dLayerConfig implements LayerConfig {
        id: ID!
        type: String!
        name: String
        num_features: Int!
        eps: Float
        momentum: Float
        affine: Boolean
        track_running_status: Boolean
    }
    type FlattenLayerConfig implements LayerConfig {
        id: ID! 
        type: String!   
        name: String    
        start_dim: Int
        end_dim: Int
    }
    type DropoutLayerConfig implements LayerConfig {
        id: ID! 
        type: String!   
        name: String    
        p: Float
    }
    type ELULayerConfig implements LayerConfig {
        id: ID! 
        type: String!   
        name: String
        alpha: Float
        inplace: Boolean
    }
    type ReLULayerConfig implements LayerConfig {
        id: ID! 
        type: String!   
        name: String    
        inplace: Boolean
    }
    type LeakyReLULayerConfig implements LayerConfig {
        id: ID! 
        type: String!   
        name: String
        negative_slope: Float    
        inplace: Boolean
    }
    type SigmoidLayerConfig implements LayerConfig {
        id: ID! 
        type: String!   
        name: String
    }
    type LogSigmoidLayerConfig implements LayerConfig {
        id: ID! 
        type: String!   
        name: String
    }
    type TanhLayerConfig implements LayerConfig {
        id: ID! 
        type: String!   
        name: String
    }
    # ---------- Layer Config Input ----------
    input LinearLayerConfigInput {
        name: String
        in_features: Int!      
        out_features: Int!
        bias: Boolean
    }
    input Conv2dLayerConfigInput {
        name: String
        in_channels: Int!
        out_channels: Int!
        kernel_size: [Int!]!
        stride: [Int]
        padding: [Int]
        dilation: [Int]
        groups: [Int]
        bias: Boolean
        padding_mode: String
    }
    input Conv1dLayerConfigInput {
        name: String
        in_channels: Int!
        out_channels: Int!
        kernel_size: [Int!]!
        stride: [Int]
        padding: [Int]
        dilation: [Int]
        groups: [Int]
        bias: Boolean
        padding_mode: String
    }
    input MaxPool2dLayerConfigInput {
        name: String
        kernel_size: [Int!]!
        stride: [Int]
        padding: [Int]
        dilation: [Int]
        return_indices: Boolean
        ceil_mode: Boolean
    }
    input MaxPool1dLayerConfigInput {
        name: String
        kernel_size: [Int!]!
        stride: [Int]
        padding: [Int]
        dilation: [Int]
        return_indices: Boolean
        ceil_mode: Boolean
    }
    input AvgPool2dLayerConfigInput {
        name: String
        kernel_size: [Int!]!
        stride: [Int]
        padding: [Int]
        count_include_pad: Boolean
        divisor_override: Int
        ceil_mode: Boolean
    }
    input AvgPool1dLayerConfigInput {
        name: String
        kernel_size: [Int!]!
        stride: [Int]
        padding: [Int]
        count_include_pad: Boolean
        ceil_mode: Boolean
    }
    input BatchNorm2dLayerConfigInput {
        name: String
        num_features: Int!
        eps: Float
        momentum: Float
        affine: Boolean
        track_running_Status: Boolean
    }
    input BatchNorm1dLayerConfigInput {
        name: String
        num_features: Int!
        eps: Float
        momentum: Float
        affine: Boolean
        track_running_Status: Boolean
    }
    input FlattenLayerConfigInput {  
        name: String    
        start_dim: Int
        end_dim: Int
    }
    input DropoutLayerConfigInput {  
        name: String    
        p: Float
    }
    input ELULayerConfigInput {
        name: String
        alpha: Float
        inplace: Boolean
    }
    input ReLULayerConfigInput {
        name: String
        inplace: Boolean
    }
    input LeakyReLULayerConfigInput {
        name: String
        negative_slope: Float    
        inplace: Boolean
    }
    input SigmoidLayerConfigInput {
        name: String
    }
    input LogSigmoidLayerConfigInput {
        name: String
    }
    input TanhLayerConfigInput {
        name: String
    }
    # LayerConfig input for collective layers
    input LayerConfigInput {
        type: String! 
        linear: LinearLayerConfigInput
        conv2d: Conv2dLayerConfigInput
        conv1d: Conv1dLayerConfigInput
        maxpool2d: MaxPool2dLayerConfigInput
        maxpool1d: MaxPool1dLayerConfigInput
        avgpool2d: AvgPool2dLayerConfigInput
        avgpool1d: AvgPool1dLayerConfigInput
        batchnorm2d: BatchNorm2dLayerConfigInput
        batchnorm1d: BatchNorm1dLayerConfigInput
        flatten: FlattenLayerConfigInput
        dropout: DropoutLayerConfigInput
        elu: ELULayerConfigInput
        relu: ReLULayerConfigInput
        leakyrelu: LeakyReLULayerConfigInput
        sigmoid: SigmoidLayerConfigInput
        logsigmoid: LogSigmoidLayerConfigInput
        tanh: TanhLayerConfigInput
    }
    # ---------- Train Config ----------
    type OptimizerConfig {
        lr: Float!
        eps: Float
        # for adam family
        weight_decay: Float
        # for adam
        betas: [Float]
        # for adadelta
        rho: Float
        # for adafactor
        beta2_decay: Float
        d: Float
        # for sgd family
        # for asgd
        lambd: Float
        alpha: Float
        t0: Float
        # for sgd
        momentum: Float
        dampening: Float
        nesterov: Boolean
        # for rprop
        etas: [Float]
        step_sizes: [Float]
        # for lgbfs
        max_iter: Int
        max_eval: Int
        tolerance_grad: Float
        tolerance_change: Float
        history_size: Int
    }
    type TrainConfig {
        epochs: Int!
        optimizer: String!
        optimizer_config: OptimizerConfig!
        loss_function: String!
    }
    type TrainStatus {
        epoch: Int!
        loss: Float!
        accuracy: Float!
        completed: Boolean!
    }
    input OptimizerConfigInput {
        lr: Float!
        eps: Float
        # for adam family
        weight_decay: Float
        # for adam
        betas: [Float]
        # for adadelta
        rho: Float
        # for adafactor
        beta2_decay: Float
        d: Float
        # for sgd family
        # for asgd
        lambd: Float
        alpha: Float
        t0: Float
        # for sgd
        momentum: Float
        dampening: Float
        nesterov: Boolean
        # for rprop
        etas: [Float]
        step_sizes: [Float]
        # for lgbfs
        max_iter: Int
        max_eval: Int
        tolerance_grad: Float
        tolerance_change: Float
        history_size: Int
    }
    input TrainConfigInput {
        epochs: Int!
        optimizer: String!
        optimizer_config: OptimizerConfigInput!
        loss_function: String!
    }
    # ---------- Dataset Config ----------
    interface DatasetConfig {
        name: String!
        batch_size: Int
        split_length: [Float]
        shuffle: Boolean
    }
    type MNISTDatasetConfig implements DatasetConfig {
        name: String!
        batch_size: Int
        split_length: [Float]
        transform: [String]
        shuffle: Boolean
        root: String!
        train: Boolean
        download: Boolean
    }
    type CIFAR10DatasetConfig implements DatasetConfig {
        name: String!
        batch_size: Int
        split_length: [Float]
        shuffle: Boolean
        transform: [String]
        root: String!
        train: Boolean
        download: Boolean
    }
    type CustomCSVDatasetConfig implements DatasetConfig {
        name: String!
        batch_size: Int
        split_length: [Float]
        shuffle: Boolean
        transform: [String]
        path_to_csv: String!
        feature_columns: [String!]!
        label_columns: [String!]!
    }
    input MNISTDatasetConfigInput {
        root: String!
        train: Boolean
        download: Boolean
        transform: [String]
    }
    input CIFAR10DatasetConfigInput {
        root: String!
        train: Boolean
        download: Boolean
        transform: [String]
    }
    input CustomCSVDatasetConfigInput {
        transform: [String]
        path_to_csv: String!
        feature_columns: [String!]!
        label_columns: [String!]!
    }
    input DatasetConfigInput {
        name: String!
        batch_size: Int
        split_length: [Float!]
        shuffle: Boolean
        mnist: MNISTDatasetConfigInput
        cifar10: CIFAR10DatasetConfigInput
        custom_csv: CustomCSVDatasetConfigInput 
    }
    # ---------- Model ----------
    interface EncoderDecoderArchitecture {
        encoder_layers_config: [LayerConfig!]!
        decoder_layers_config: [LayerConfig!]!
    }
    enum VAELayerTarget {
        Encoder,
        Decoder,
        Mean,
        Logvar
    }
    type VAEArchitecture implements EncoderDecoderArchitecture {
        encoder_layers_config: [LayerConfig!]!
        decoder_layers_config: [LayerConfig!]!
        mean_layers_config: [LayerConfig!]
        log_var_layers_config: [LayerConfig!]
        latent_dim: Int!
    }
    input VAEArchitectureInput {
        latent_dim: Int!
        target: VAELayerTarget!
    }
    input SpecialModelArchitectureInput {
        vae: VAEArchitectureInput
    }
    type SpecialModelArchitecture {
        vae: VAEArchitecture
    }
    # Model type
    type Model {
        id: ID!
        name: String!       
        layers_config: [LayerConfig!]! 
        train_config: TrainConfig!
        dataset_config: DatasetConfig!
        special: SpecialModelArchitecture
    }
    type ModelDimensionResolveStatusStruct {
        layer_id: ID!
        message: String!
    }
    type ModelDimensionResolveStatus {
        status: [ModelDimensionResolveStatusStruct]
    }
    # ---------- Queries ----------
    type Query {
        # get the model by id
        getModel(id: ID!): Model!
        # get all models
        getModels: [Model!]!
        # get training status
        getTrainingStatus: TrainStatus!
        # validate model
        validateModel(id: ID!, in_dimension: [Int!]!): ModelDimensionResolveStatus!
    }
    # ---------- Mutations ----------
    type Mutation {
        # create model
        createModel(name: String!): Model!
        # append layer to model
        appendLayer(
            model_id: ID!
            layer_config: LayerConfigInput!
            special: SpecialModelArchitectureInput
        ): Model!
        # delete layer
        deleteLayer(
            model_id: ID!, 
            layer_id: ID!): Model!
        # set's training configuration
        setTrainConfig(
            model_id: ID!
            train_config: TrainConfigInput!
        ): Model!
        # set's dataset configuration
        setDataset(
            model_id: ID!
            dataset_config: DatasetConfigInput!
        ): Model!
        # train model
        train(
            model_id: ID!
        ): Model!
        # save model
        save: Boolean
        # load model
        load: [Model]!
        # start tensorboard
        startTensorboard : String!
    }
`

