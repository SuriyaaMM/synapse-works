import { graphStore } from './graphStore.svelte';

export function createStorageService(modelId: string) {
  
  function saveStateToStorage() {
    if (typeof window === 'undefined' || !modelId) return;
    
    try {
      const state = graphStore.getStateForStorage();
      const stateWithTimestamp = {
        ...state,
        timestamp: Date.now(),
        version: '1.0' // For future migration compatibility
      };
      
      sessionStorage.setItem(`graph-state-${modelId}`, JSON.stringify(stateWithTimestamp));
      console.log('Graph state saved to storage');
    } catch (error) {
      console.error('Error saving graph state to storage:', error);
    }
  }

  function loadStateFromStorage() {
    if (typeof window === 'undefined' || !modelId) return;
    
    try {
      const saved = sessionStorage.getItem(`graph-state-${modelId}`);
      if (saved) {
        const state = JSON.parse(saved);
        
        // Validate that the state has the expected structure
        if (state && typeof state === 'object' && state.nodes && state.edges) {
          graphStore.loadStateFromStorage(state);
          console.log('Graph state loaded from storage');
        } else {
          console.warn('Invalid saved state structure, clearing storage');
          sessionStorage.removeItem(`graph-state-${modelId}`);
        }
      }
    } catch (error) {
      console.error('Error loading saved state:', error);
      // Clear corrupted data
      sessionStorage.removeItem(`graph-state-${modelId}`);
    }
  }

  function clearStorage() {
    if (typeof window === 'undefined' || !modelId) return;
    
    try {
      sessionStorage.removeItem(`graph-state-${modelId}`);
      console.log('Graph state cleared from storage');
    } catch (error) {
      console.error('Error clearing storage:', error);
    }
  }

  return {
    saveStateToStorage,
    loadStateFromStorage,
    clearStorage
  };
}