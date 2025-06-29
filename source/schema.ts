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
        bias: Boolean
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
    type ConvTranspose2dLayerConfig implements LayerConfig {
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
        output_padding: [Int]
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
        divisor_override: Int
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
    type Dropout2dLayerConfig implements LayerConfig {
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
    input ConvTranspose2dLayerConfigInput {
        name: String
        in_channels: Int!
        out_channels: Int!
        kernel_size: [Int!]!
        stride: [Int]
        padding: [Int]
        dilation: [Int]
        groups: [Int]
        bias: Boolean
        output_padding: [Int]
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
        divisor_override: Int
        ceil_mode: Boolean
    }
    input BatchNorm2dLayerConfigInput {
        name: String
        num_features: Int!
        eps: Float
        momentum: Float
        affine: Boolean
        track_running_status: Boolean
    }
    input BatchNorm1dLayerConfigInput {
        name: String
        num_features: Int!
        eps: Float
        momentum: Float
        affine: Boolean
        track_running_status: Boolean
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
    input Dropout2dLayerConfigInput {  
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
        convtranspose2d: ConvTranspose2dLayerConfigInput
        conv1d: Conv1dLayerConfigInput
        maxpool2d: MaxPool2dLayerConfigInput
        maxpool1d: MaxPool1dLayerConfigInput
        avgpool2d: AvgPool2dLayerConfigInput
        avgpool1d: AvgPool1dLayerConfigInput
        batchnorm2d: BatchNorm2dLayerConfigInput
        batchnorm1d: BatchNorm1dLayerConfigInput
        flatten: FlattenLayerConfigInput
        dropout: DropoutLayerConfigInput
        dropout2d: Dropout2dLayerConfigInput
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
        root: String!
        feature_columns: [String!]!
        label_columns: [String!]!
        is_regression_task: Boolean
    }
    type ImageFolderDatasetConfig implements DatasetConfig {
        name: String!
        batch_size: Int
        split_length: [Float]
        shuffle: Boolean
        root: String!
        transform: [String]
        allow_empty: Boolean
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
        root: String!
        feature_columns: [String!]!
        label_columns: [String!]!
        is_regression_task: Boolean
    }
    input ImageFolderDatasetConfigInput {
        root: String!
        transform: [String]
        allow_empty: Boolean
    }
    input DatasetConfigInput {
        name: String!
        batch_size: Int
        split_length: [Float!]
        shuffle: Boolean
        mnist: MNISTDatasetConfigInput
        cifar10: CIFAR10DatasetConfigInput
        image_folder: ImageFolderDatasetConfigInput
        custom_csv: CustomCSVDatasetConfigInput
    }
    # ---------- Model ----------
    type ModuleAdjacencyList {
        source_id: ID!
        target_ids: [ID!]!
    }
    type ModuleGraph {
        layers: [LayerConfig!]!
        edges: [ModuleAdjacencyList!]!
        sorted: [String!]!
    }
    input ModuleAdjacencyListInput {
        source_id: ID!
        target_ids: [ID!]!
    }
    input ModuleGraphInput {
        layers: [LayerConfigInput!]!
        edges: [ModuleAdjacencyListInput!]!
    }
    # Model type
    type Model {
        id: ID!
        name: String!
        # this is optional for now, but in future we should make layers_config optional       
        module_graph: ModuleGraph
        layers_config: [LayerConfig!]! 
        train_config: TrainConfig!
        dataset_config: DatasetConfig!
    }
    type ModelDimensionResolveStatusStruct {
        layer_id: ID!
        message: String
        in_dimension: [Int!]!
        out_dimension: [Int!]!
        required_in_dimension: [Int!]
    }
    type ModelDimensionResolveStatus {
        status: [ModelDimensionResolveStatusStruct]
    }
    # ---------- Args ------------
    enum ExportType {
        TorchTensor
        ONNX
    }
    input TrainArgs {
        export_to: ExportType
    }
    # ---------- Queries ----------
    type Query {
        # get the model by id
        getModel(id: ID!): Model!
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
        ): Model!
        # delete layer
        deleteLayer(
            model_id: ID!, 
            layer_id: ID!): Model!
        # modify layer
        modifyLayer(
            model_id: ID!,
            layer_id: ID!,
            layer_config: LayerConfigInput!): Model!
        # appends layerconfig to module graph
        appendToModuleGraph(
            layer_config: LayerConfigInput!): ModuleGraph!
        # deletes layer_id in the graph
        deleteInModuleGraph(
            layer_id: ID!) : ModuleGraph!
        # adds an edge
        connectInModuleGraph(
            source_layer_id: ID!
            target_layer_id: ID!) : ModuleGraph!
        # removes an edge
        disconnectInModuleGraph(
            source_layer_id: ID!
            target_layer_id: ID!) : ModuleGraph!
        # build's the graph and sorts it topologically & sets it to the model
        buildModuleGraph: Model!
        # set's training configuration
        setTrainConfig(
            model_id: ID!
            train_config: TrainConfigInput!): Model!
        # set's dataset configuration
        setDataset(
            model_id: ID!
            dataset_config: DatasetConfigInput!): Model!
        # train model
        train(
            model_id: ID!
            args: TrainArgs): Model!
        # save model
        saveModel: Boolean!
        # load model
        loadModel(model_id: String!): Model!
        # start tensorboard
        startTensorboard: String!
    }
`

