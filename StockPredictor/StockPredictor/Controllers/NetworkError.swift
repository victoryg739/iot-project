// NetworkError.swift
// Error types for network operations

import Foundation

// Error type for network operations
enum NetworkError: Error {
    case invalidURL
    case invalidResponse
    case invalidData
    case serverError(String)
    
    var errorDescription: String {
        switch self {
        case .invalidURL:
            return "Invalid server URL"
        case .invalidResponse:
            return "Invalid response from server"
        case .invalidData:
            return "Couldn't decode server response"
        case .serverError(let message):
            return message
        }
    }
}
