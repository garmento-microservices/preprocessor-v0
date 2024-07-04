CREATE TABLE preprocessing_jobs (
    id VARCHAR(64) PRIMARY KEY,
    status VARCHAR(30),
    ref_image VARCHAR(1024) NOT NULL,
    garment_image VARCHAR(1024) NOT NULL,
    masked_garment_image VARCHAR(1024) DEFAULT NULL,
    densepose_image VARCHAR(1024) DEFAULT NULL,
    segmented_image VARCHAR(1024) DEFAULT NULL,
    pose_keypoints VARCHAR(1024) DEFAULT NULL
);
