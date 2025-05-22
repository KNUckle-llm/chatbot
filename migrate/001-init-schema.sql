-- knuckle 데이터베이스 스키마 초기화
CREATE DATABASE IF NOT EXISTS knuckle;
USE knuckle;

SET NAMES utf8mb4;
SET CHARACTER_SET_CLIENT = utf8mb4;

-- user 테이블 생성
CREATE TABLE IF NOT EXISTS `user` (
  `user_id` varchar(45) NOT NULL,
  `pw` varchar(45) DEFAULT NULL,
  `name` varchar(45) DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`user_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- session 테이블 생성
CREATE TABLE IF NOT EXISTS `session` (
  `session_id` varchar(45) NOT NULL,
  `user_id` varchar(45) DEFAULT NULL,
  `started_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `ended_at` timestamp NULL DEFAULT NULL,
  PRIMARY KEY (`session_id`),
  KEY `user_id_idx` (`user_id`),
  CONSTRAINT `user_id` FOREIGN KEY (`user_id`) REFERENCES `user` (`user_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- message 테이블 생성
CREATE TABLE IF NOT EXISTS `message` (
  `message_id` varchar(45) NOT NULL,
  `session_id` varchar(45) DEFAULT NULL,
  `role` varchar(45) DEFAULT NULL,
  `content` longtext,
  `sent_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`message_id`),
  KEY `session_id_idx` (`session_id`),
  CONSTRAINT `session_id` FOREIGN KEY (`session_id`) REFERENCES `session` (`session_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- 초기 데이터 삽입
INSERT INTO `user` VALUES ('knuckle','knuckle','캡스톤','2025-05-21 02:30:49') 
ON DUPLICATE KEY UPDATE `name`=VALUES(`name`);