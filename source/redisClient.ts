import Redis from 'ioredis'; 
import type { Redis as RedisClientType } from 'ioredis';

const REDIS_URL = process.env.REDIS_URL || 'redis://localhost:6379';
const REDIS_QUEUE_NAME = 'model_layer_updates_queue';

let redisInstance: RedisClientType | null = null;

// export for use in index.ts
export function connectRedis() {

    // if already connected return the object
    if (redisInstance) {
        console.log("Redis client already connected.");
        return redisInstance;
    }
    try {
        // create new redis object
        redisInstance = new Redis(REDIS_URL);
        // connect to redis-server
        redisInstance.on('connect', () => {
            console.log("[synapse][redis]: Connected to Redis.");
        });
        // handle error
        redisInstance.on('error', (err) => {
            console.error("[synapse][redis]: Redis connection error:", err);
        });

        return redisInstance;
    } catch (error) {
        console.error("[synapse][redis]: Failed to connect to Redis:", error);
        throw error;
    }
}

export async function enqueueMessage(message: object) {
    if (!redisInstance) {
        console.error("[synapse][redis]: Redis client not initialized. Message not enqueued:", message);
        return false;
    }
    try {
        // stringify message
        const messageString = JSON.stringify(message);
        // push to messgae queue
        const result = await redisInstance.lpush(REDIS_QUEUE_NAME, messageString);
        console.log(`[synapse][redis]: Message enqueued to '${REDIS_QUEUE_NAME}'. List length: ${result}`);
        return true;
    } catch (error) {
        console.error(`[synapse][redis]: Failed to enqueue message to '${REDIS_QUEUE_NAME}':`, error);
        return false;
    }
}