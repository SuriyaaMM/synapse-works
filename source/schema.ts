export const typeDefs = `#graphql
    # abstract Layer type
    interface Layer {
        id: ID! 
        type: String! # layer name (Linear, Conv2d)
        name: String # optional name
    }
    # LinearLayer type derived from abstract Layer
    type LinearLayer implements Layer {
        id: ID! 
        type: String!   # must be Linear
        name: String    # optional layer name
        inputDim: Int!  # input dimensions
        outputDim: Int! # output dimensions
    }
    # Model type
    type Model {
        id: ID!
        name: String!       # model name
        layers: [Layer!]!   # list of layers that model contains
    }
    # LinearLayerConfig input 
    input LinearLayerConfig {
        inputDim: Int!      
        outputDim: Int!
        name: String
    }
    # LayerConfig input for collective layers
    input LayerConfig {
        type: String! # Linear, Conv2D
        linear: LinearLayerConfig # optional LinearLayerConfig
    }
    # Queries
    type Query {
        # get the model by id
        getModel(id: ID!): Model
        # get all models
        getModels: [Model!]!
    }
    # Mutations
    type Mutation {
        # create model
        createModel(name: String!): Model!
        # append layer to model
        appendLayer(
            modelId: ID!
            layerConfig: LayerConfig!
        ): Model!
    }
`

