import { graphStore } from './graphStore.svelte';

export function createStorageService(modelId: string) {
  
  function saveStateToStorage() {
    if (typeof window === 'undefined' || !modelId) return;
    
    try {
      const state = graphStore.getStateForStorage();
      const stateWithTimestamp = {
        ...state,
        timestamp: Date.now(),
        version: '1.0',
        modelId: modelId 
      };
      
      const key = `graph-state-${modelId}`;
      sessionStorage.setItem(key, JSON.stringify(stateWithTimestamp));
      console.log(`Graph state saved to storage for model: ${modelId}`);
    } catch (error) {
      console.error('Error saving graph state to storage:', error);
    }
  }

  function loadStateFromStorage() {
    if (typeof window === 'undefined' || !modelId) return;
    
    try {
      const key = `graph-state-${modelId}`;
      const saved = sessionStorage.getItem(key);
      
      if (saved) {
        const state = JSON.parse(saved);
        
        // Validate that the state has the expected structure
        if (isValidState(state) && state.modelId === modelId) {
          graphStore.loadStateFromStorage(state);
          console.log(`Graph state loaded from storage for model: ${modelId}`);
          return true;
        } else {
          console.warn(`Invalid saved state structure for model ${modelId}, clearing storage`);
          sessionStorage.removeItem(key);
        }
      } else {
        console.log(`No saved state found for model: ${modelId}, starting with blank canvas`);
      }
      
      // If no valid state found, initialize with blank canvas
      initializeBlankCanvas();
      return false;
      
    } catch (error) {
      console.error('Error loading saved state:', error);
      // Clear corrupted data and initialize blank canvas
      sessionStorage.removeItem(`graph-state-${modelId}`);
      initializeBlankCanvas();
      return false;
    }
  }

  function isValidState(state: any): boolean {
    return (
      state &&
      typeof state === 'object' &&
      Array.isArray(state.nodes) &&
      Array.isArray(state.edges) &&
      typeof state.timestamp === 'number' &&
      typeof state.version === 'string'
    );
  }

  function initializeBlankCanvas() {
    graphStore.resetToInitialState();
    console.log(`Initialized blank canvas for model: ${modelId}`);
  }

  function clearStorage() {
    if (typeof window === 'undefined' || !modelId) return;
    
    try {
      const key = `graph-state-${modelId}`;
      sessionStorage.removeItem(key);
      console.log(`Graph state cleared from storage for model: ${modelId}`);
      initializeBlankCanvas();
    } catch (error) {
      console.error('Error clearing storage:', error);
    }
  }

  function hasStoredState(): boolean {
    if (typeof window === 'undefined' || !modelId) return false;
    
    try {
      const key = `graph-state-${modelId}`;
      const saved = sessionStorage.getItem(key);
      return saved !== null;
    } catch (error) {
      return false;
    }
  }

  function getStoredModelIds(): string[] {
    if (typeof window === 'undefined') return [];
    
    try {
      const keys = Object.keys(sessionStorage);
      return keys
        .filter(key => key.startsWith('graph-state-'))
        .map(key => key.replace('graph-state-', ''));
    } catch (error) {
      return [];
    }
  }

  return {
    saveStateToStorage,
    loadStateFromStorage,
    clearStorage,
    hasStoredState,
    getStoredModelIds,
    initializeBlankCanvas
  };
}