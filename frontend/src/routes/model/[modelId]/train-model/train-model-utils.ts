import client from '$lib/apolloClient';
import { TRAIN_MODEL } from '$lib/mutations';
import { GET_TRAINING_STATUS } from '$lib/queries';
import type { TrainStatus } from '../../../../../../source/types/trainTypes';
import type { GraphQLTrainArgs, TrainArgs } from '../../../../../../source/types/argTypes';
import { ExportType } from '../../../../../../source/types/argTypes';

export class TrainingUtils {
  private statusInterval: any = null;
  private onStatusUpdate: ((status: TrainStatus | null) => void) | null = null;
  private onTrainingComplete: (() => void) | null = null;
  private onError: ((error: string) => void) | null = null;

  constructor() {}

  setCallbacks(
    onStatusUpdate: (status: TrainStatus | null) => void,
    onTrainingComplete: () => void,
    onError: (error: string) => void
  ) {
    this.onStatusUpdate = onStatusUpdate;
    this.onTrainingComplete = onTrainingComplete;
    this.onError = onError;
  }

  async checkTrainingStatus(): Promise<TrainStatus | null> {
    try {
      const response = await client.query({
        query: GET_TRAINING_STATUS,
        fetchPolicy: 'no-cache'
      });
      
      const newStatus = response.data?.getTrainingStatus;
      
      if (newStatus) {
        this.onStatusUpdate?.(newStatus);
        
        if (!newStatus.completed && !this.statusInterval) {
          this.startStatusPolling();
        }
        
        return newStatus;
      }
      
      return null;
    } catch (err) {
      console.error('Error checking training status:', err);
      this.onError?.('Failed to check training status');
      return null;
    }
  }


  async startTraining(exportType: ExportType): Promise<boolean> {
    try {
      const args: GraphQLTrainArgs = {
        export_to: exportType
      };

      const variables: TrainArgs = { args };

      const res = await client.mutate({
        mutation: TRAIN_MODEL,
        variables,
        errorPolicy: 'all'
      });
      
      if (res.errors && res.errors.length > 0) {
        throw new Error(`GraphQL Error: ${res.errors.map(e => e.message).join(', ')}`);
      }
      
      if (!res.data?.train) {
        throw new Error('Failed to start training - no data returned');
      }
      
      // Initialize training status
      const initialStatus: TrainStatus = {
        epoch: 0,
        completed: false,
        loss: 0,
        accuracy: 0,
        started: true
      };
      
      this.onStatusUpdate?.(initialStatus);
      this.startStatusPolling();
      
      // Check for immediate updates
      await this.checkImmediateStatus();
      
      return true;
      
    } catch (err: any) {
      console.error('Training Error Details:', {
        message: err.message,
        networkError: err.networkError,
        graphQLErrors: err.graphQLErrors,
        fullError: err
      });
      
      let errorMessage = 'Unknown error occurred';
      
      if (err.networkError) {
        errorMessage = `Network Error: ${err.networkError.message}`;
      } else if (err.graphQLErrors && err.graphQLErrors.length > 0) {
        errorMessage = `GraphQL Error: ${err.graphQLErrors.map((e: any) => e.message).join(', ')}`;
      } else {
        errorMessage = err.message || err.toString();
      }
      
      this.onError?.(errorMessage);
      return false;
    }
  }


  stopTraining(): void {
    if (this.statusInterval) {
      clearInterval(this.statusInterval);
      this.statusInterval = null;
    }

    console.log('Training stopped by user');
  }


  private startStatusPolling(): void {
    if (this.statusInterval) {
      clearInterval(this.statusInterval);
    }
    
    this.statusInterval = setInterval(async () => {
      try {
        const response = await client.query({
          query: GET_TRAINING_STATUS,
          fetchPolicy: 'no-cache'
        });
        
        const newStatus = response.data?.getTrainingStatus;
        
        if (newStatus) {
          this.onStatusUpdate?.(newStatus);
          
          // Stop polling if training is completed
          if (newStatus.completed) {
            this.stopPolling();
            this.onTrainingComplete?.();
          }
        }
        
      } catch (err) {
        console.error('Error polling training status:', err);
        // Don't stop polling on error, just log it
      }
    }, 1000); // Poll every 1 second for real-time updates
  }

  private async checkImmediateStatus(): Promise<void> {
    for (let i = 0; i < 10; i++) {
      await new Promise(resolve => setTimeout(resolve, 200 * (i + 1))); 
      await this.checkTrainingStatus();
    }
  }

  private stopPolling(): void {
    if (this.statusInterval) {
      clearInterval(this.statusInterval);
      this.statusInterval = null;
    }
  }


  cleanup(): void {
    this.stopPolling();
  }

  static validateModelConfig(modelDetails: any): string | null {
    if (!modelDetails?.module_graph?.layers.length) {
      return 'Model has no module graph defined';
    }

    if (!modelDetails?.train_config) {
      return 'Model has no training configuration';
    }

    if (!modelDetails?.dataset_config) {
      return 'Model has no dataset configuration';
    }

    return null;
  }
}