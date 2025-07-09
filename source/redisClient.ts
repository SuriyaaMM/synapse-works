import Redis from 'ioredis'; 
import type { Redis as RedisClientType } from 'ioredis';
import { readFile } from 'fs/promises';
import { TrainStatus } from './types/trainTypes.js'

let REDIS_URL = 'redis://localhost:6379';
const REDIS_MAIN_QUEUE_NAME = 'model_main_queue';
const REDIS_TRAIN_QUEUE_NAME = 'model_train_queue';
export let REMOTE_HOSTING = false;
export let PORT: number = 4000;

let redis_instance: RedisClientType | null = null;

// load environment.json
const file_path = "environment.json";
let ENVIRONMENT = await readFile(file_path, 'utf-8');
ENVIRONMENT = JSON.parse(ENVIRONMENT);

async function configureRedisURL(){
    // skip default arguements
    const args = process.argv.slice(2);
    let remote: boolean = false; 
    
    for(const arg of args){
        if(arg === "--remote"){
            remote = true;
            REMOTE_HOSTING = true;
        }
    }
    // modify REDIS URL on remote
    if(remote){
        REDIS_URL = ENVIRONMENT["REDIS_URL"];
    }
}

// export for use in index.ts
export async function connectRedis() {
    await configureRedisURL();
    console.log(`[synapse][redis]: Connected to redis: ${REDIS_URL}`)

    // if already connected return the object
    if (redis_instance) {
        console.log("Redis client already connected.");
        return redis_instance;
    }
    try {
        // create new redis object
        redis_instance = new Redis(REDIS_URL);
        // connect to redis-server
        redis_instance.on('connect', () => {
            console.log("[synapse][redis]: Connected to Redis.");
        });
        // handle error
        redis_instance.on('error', (err) => {
            console.error("[synapse][redis]: Redis connection error:", err);
        });

        return redis_instance;
    } catch (error) {
        console.error("[synapse][redis]: Failed to connect to Redis:", error);
        throw error;
    }
}

export async function enqueueMessage(message: object) {
    if (!redis_instance) {
        console.error("[synapse][redis]: Redis client not initialized. Message not enqueued:", message);
        return false;
    }
    try {
        // stringify message
        const messageString = JSON.stringify(message);
        // push to messgae queue
        const result = await redis_instance.lpush(REDIS_MAIN_QUEUE_NAME, messageString);
        console.log(`[synapse][redis]: Message enqueued to '${REDIS_MAIN_QUEUE_NAME}'. List length: ${result}`);
        return true;
    } catch (error) {
        console.error(`[synapse][redis]: Failed to enqueue message to '${REDIS_MAIN_QUEUE_NAME}':`, error);
        return false;
    }
}
export async function dequeueMessage() {
    if (!redis_instance) {
        console.error("[synapse][redis]: Redis client not initialized");
        return false;
    }
    try {
        const result = await redis_instance.rpop(REDIS_TRAIN_QUEUE_NAME);
        console.log(`result = ${result}`);

        if(result){
            let message : TrainStatus = JSON.parse(result.toString());
            console.log(`received status: ${JSON.stringify(message)}`);

            return message;
        }
    } catch (error) {
        console.error(`[synapse][redis]: Failed to dequeue message to '${REDIS_TRAIN_QUEUE_NAME}':`, error);
        return false;
    }
}