export interface DatasetConfig {
    name: string;
    batch_size?: number;
    split_length?: number[];
    shuffle?: boolean;
};

export interface MNISTDatasetConfig extends DatasetConfig {
    root: string;
    train?: boolean;
    download?: boolean;
    transform?: string[];
};

export interface CIFAR10DatasetConfig extends DatasetConfig {
    root: string;
    train?: boolean;
    download?: boolean;
    transform?: string[];
};

export interface CelebADatasetConfig extends DatasetConfig {
    root: string;
    target_type?: string[];
    download?: boolean;
    transform?: string[];
    target_transform?: string[];
};

export interface VOCSegmentationDatasetConfig extends DatasetConfig  {
    root: string;
    image_set?: string;
    year?: string;
    download?: boolean;
    transform?: string[];
    target_transform?: string[];
};

export interface CustomCSVDatasetConfig extends DatasetConfig {
    root: string;
    feature_columns: string[];
    label_columns: string[];
    is_regression_task: boolean;
}

export interface ImageFolderDatasetConfig extends DatasetConfig {
    root: string;
    transform?: string[];
    allow_empty?: boolean;
}

export type MNISTDatasetConfigInput = {
    root: string;
    train?: boolean;
    download?: boolean;
    transform?: string[];
};

export type CIFAR10DatasetConfigInput = {
    root: string;
    train?: boolean;
    download?: boolean;
    transform?: string[];
};

export type CelebADatasetConfigInput =  {
    root: string;
    target_type?: string[];
    download?: boolean;
    transform?: string[];
    target_transform?: string[];
};

export type VOCSegmentationDatasetConfigInput =  {
    root: string;
    image_set?: string;
    year?: string;
    download?: boolean;
    transform?: string[];
    target_transform?: string[];
};

export type CustomCSVDatasetConfigInput = {
    root: string;
    feature_columns: string[];
    label_columns: string[];
    is_regression_task?: boolean;
}

export type ImageFolderDatasetConfigInput = {
    root: string;
    transform?: string[];
    allow_empty?: boolean;
}

export type DatasetConfigInput  = {
    name: string;
    batch_size?: number;
    split_length?: number[];
    shuffle?: boolean;
    mnist?: MNISTDatasetConfigInput;
    cifar10?: CIFAR10DatasetConfigInput;
    celeba?: CelebADatasetConfigInput;
    vocsegmentation?: VOCSegmentationDatasetConfigInput;
    image_folder?: ImageFolderDatasetConfigInput;
    custom_csv?: CustomCSVDatasetConfigInput;
};