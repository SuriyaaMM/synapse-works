import client from '$lib/apolloClient';
import type { LayerConfigInput } from '../../../../../../source/types/layerTypes';
import { ADD_TO_GRAPH, CONNECT_NODES, DELETE_FROM_GRAPH, DISCONNECT_NODES, BUILD_MODULE_GRAPH, VALIDATE_GRAPH } from '$lib/mutations';
import { tick } from 'svelte';
import { fetchModelDetails, modelDetails } from '../modelDetails';
import { graphStore } from './graphStore.svelte';
import { get } from 'svelte/store';

export function createGraphMutations(
  saveStateToStorage: () => void,
  dispatch: (eventName: string, detail?: any) => void
) {
  let validationTrigger = 0;

  async function handleAddLayer(event: CustomEvent<{ layerConfig: LayerConfigInput }>) {
    const { layerConfig } = event.detail;
    

    graphStore.setLoading(true);
    graphStore.setError(null);

    try {
      console.log("just before sending to mutation:", layerConfig);
      const response = await client.mutate({
        mutation: ADD_TO_GRAPH,
        variables: { layer_config: layerConfig },
        fetchPolicy: 'no-cache'
      });
      console.log(response);

      if (!response.data?.appendToModuleGraph) {
        throw new Error('Failed to add layer to graph - no data returned');
      }

      const graphData = response.data.appendToModuleGraph;
      graphStore.updateNodesFromGraphData(graphData);
      graphStore.incrementNodeCounter();

      saveStateToStorage();
      dispatch('layerAdded', { layerConfig }); 

    } catch (err) {
      console.error('Error adding layer to graph:', err);
      graphStore.setError(err instanceof Error ? err.message : 'Unknown error occurred while adding layer');
    } finally {
      graphStore.setLoading(false);
    }
  }

  async function createConnection(fromId: string, toId: string) {
    graphStore.setLoading(true);
    graphStore.setError(null);

    try {
      const response = await client.mutate({
        mutation: CONNECT_NODES,
        variables: { source_layer_id: fromId, target_layer_id: toId },
        fetchPolicy: 'no-cache'
      });

      if (!response.data?.connectInModuleGraph) {
        throw new Error('Failed to connect layers - no data returned');
      }

      const graphData = response.data.connectInModuleGraph;
      graphStore.updateNodesFromGraphData(graphData);

      dispatch('connectionCreated', { fromId, toId });
      saveStateToStorage();

    } catch (err) {
      console.error('Error creating connection:', err);
      graphStore.setError(err instanceof Error ? err.message : 'Unknown error occurred while connecting layers');
    } finally {
      graphStore.setLoading(false);
    }
  }

  async function handleDeleteNode() {
    if (!graphStore.selectedNode) {
      graphStore.setError('No layer selected for deletion');
      return;
    }

    graphStore.setLoading(true);
    graphStore.setError(null);

    try {
      const response = await client.mutate({
        mutation: DELETE_FROM_GRAPH,
        variables: { layer_id: graphStore.selectedNode.id },
        fetchPolicy: 'no-cache'
      });

      if (!response.data?.deleteInModuleGraph) {
        throw new Error('Failed to delete layer - no data returned');
      }

      const graphData = response.data.deleteInModuleGraph;
      graphStore.updateNodesFromGraphData(graphData);

      dispatch('layerDeleted', { layerId: graphStore.selectedNode.id });
      graphStore.setSelectedNode(null);
      saveStateToStorage();

    } catch (err) {
      console.error('Error deleting layer:', err);
      graphStore.setError(err instanceof Error ? err.message : 'Unknown error occurred while deleting layer');
    } finally {
      graphStore.setLoading(false);
    }
  }

  async function handleDeleteEdge() {
    if (!graphStore.selectedEdge) {
      console.warn('No edge selected for deletion');
      return;
    }

    graphStore.setLoading(true);
    graphStore.setError(null);

    try {
      const response = await client.mutate({
        mutation: DISCONNECT_NODES,
        variables: {
          source_layer_id: graphStore.selectedEdge.source,
          target_layer_id: graphStore.selectedEdge.target
        },
        fetchPolicy: 'no-cache'
      });

      if (!response.data?.disconnectInModuleGraph) {
        throw new Error('Failed to disconnect layers - no data returned');
      }

      const graphData = response.data.disconnectInModuleGraph;
      graphStore.updateNodesFromGraphData(graphData);

      dispatch('connectionDeleted', { connectionId: graphStore.selectedEdge.id });
      graphStore.setSelectedEdge(null);
      saveStateToStorage();

    } catch (err) {
      console.error('Error disconnecting layers:', err);
      graphStore.setError(err instanceof Error ? err.message : 'Unknown error occurred while disconnecting layers');
    } finally {
      graphStore.setLoading(false);
    }
  }

  async function buildModuleGraph() {
    graphStore.setLoading(true);
    graphStore.setError(null);

    try {
      const response = await client.mutate({
        mutation: BUILD_MODULE_GRAPH,
        variables: {},
        fetchPolicy: 'no-cache'
      });

      console.log('hii', response);

      if (!response.data?.buildModuleGraph) {
        throw new Error('Failed to build module graph - no data returned');
      }

      const graphData = response.data.buildModuleGraph;
      graphStore.setBuildResult(graphData);

      await validateGraphStructure();

      if (graphData.module_graph) {
        graphStore.updateNodesFromGraphData(graphData.module_graph);
      }

      saveStateToStorage();

    } catch (err) {
      console.error('Error building module graph:', err);
      graphStore.setError(err instanceof Error ? err.message : 'Unknown error occurred while building module graph');
    } finally {
      graphStore.setLoading(false);
    }
  }

  async function validateGraphStructure() {
    fetchModelDetails();
    try {
      let inputDimension: number[];
      const model = get(modelDetails);
      const datasetType = model?.dataset_config?.name?.toLowerCase();

      switch (datasetType) {
      case 'mnist':
        inputDimension = [1, 28, 28];
        break;
      case 'cifar10':
        inputDimension = [3, 32, 32];
        break;
      case 'image_folder':
        inputDimension = [3, 224, 224];
        break;
      case 'celeba':
        inputDimension = [3, 224, 224]; // Common preprocessing size
        break;
      case 'voc_segmentation':
        inputDimension = [3, 224, 224]; // Common input size for segmentation tasks
        break;
      default:
        inputDimension = [1, 28, 28];
    }

    console.log(inputDimension);
      const response = await client.mutate({
        mutation: VALIDATE_GRAPH,
        variables: { in_dimension: inputDimension },
        fetchPolicy: 'no-cache',
        errorPolicy: 'all'
      });

      if (response.data?.validateModuleGraph) {
        graphStore.setGraphValidationResult({
          ...response.data.validateModuleGraph,
          status: response.data.validateModuleGraph.status || []
        });
      } else {
        graphStore.setGraphValidationResult(null);
      }

      validationTrigger++;
      await tick();

    } catch (err) {
      console.error('Validation error:', err);
      graphStore.setGraphValidationResult(null);
      validationTrigger++;
    }
  }

  return {
    handleAddLayer,
    createConnection,
    handleDeleteNode,
    handleDeleteEdge,
    buildModuleGraph,
    validateGraphStructure
  };
}
