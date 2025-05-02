// AppConfig.swift
// Application configuration

import Foundation

struct AppConfig {
    // Replace with your Flask backend URL
    static let serverBaseURL = "http://127.0.0.1:8010"
    
    // For simulator testing, use your computer's local IP, for example:
    // static let serverBaseURL = "http://192.168.1.100:5001"
    
    // For physical device testing with VM on GCP, use the public IP address:
    // static let serverBaseURL = "http://34.123.456.789:5001"
}
