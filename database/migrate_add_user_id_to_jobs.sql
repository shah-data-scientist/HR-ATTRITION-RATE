-- Migration: Add user_id column to jobs table
-- This allows tracking which user created each async job

ALTER TABLE jobs 
ADD COLUMN IF NOT EXISTS user_id VARCHAR(50) NOT NULL DEFAULT 'demo1';

-- Update existing records to use default user_id if needed
UPDATE jobs 
SET user_id = 'demo1' 
WHERE user_id IS NULL;
