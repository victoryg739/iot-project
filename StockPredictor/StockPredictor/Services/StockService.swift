// StockService.swift
// Service for handling API communication

import Foundation

// Protocol for dependency injection and testing
protocol StockServiceProtocol {
    func getPrediction(for ticker: String, forceRetrain: Bool) async throws -> StockPrediction
    func submitModelUpdate(ticker: String, date: String, actualPrice: Double) async throws -> Bool
}

// Service implementation for fetching stock predictions
class StockService: StockServiceProtocol {
    private let baseURL: String
    
    init(baseURL: String? = nil) {
        // Use provided URL or default to the app config
        self.baseURL = baseURL ?? AppConfig.serverBaseURL
    }
    
    func getPrediction(for ticker: String, forceRetrain: Bool = false) async throws -> StockPrediction {
        guard let url = URL(string: "\(baseURL)/predict") else {
            throw NetworkError.invalidURL
        }
        
        // Log request for debugging in development
        #if DEBUG
        print("Making request to: \(url.absoluteString)")
        #endif
        
        // Create request body
        let body: [String: Any] = [
            "ticker": ticker,
            "force_retrain": forceRetrain
        ]
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        
        // Perform the request
        let (data, response) = try await URLSession.shared.data(for: request)
        
        // Check response status
        guard let httpResponse = response as? HTTPURLResponse else {
            throw NetworkError.invalidResponse
        }
        
        // Handle server errors
        if httpResponse.statusCode >= 400 {
            // Try to parse error message
            if let errorObject = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let errorMessage = errorObject["error"] as? String {
                throw NetworkError.serverError(errorMessage)
            } else {
                throw NetworkError.serverError("Server returned status code \(httpResponse.statusCode)")
            }
        }
        
        // Decode the response
        do {
            let decoder = JSONDecoder()
            let prediction = try decoder.decode(StockPrediction.self, from: data)
            return prediction
        } catch {
            #if DEBUG
            print("Decoding error: \(error)")
            #endif
            
            // More detailed decoding error handling in development
            if let decodingError = error as? DecodingError {
                switch decodingError {
                case .keyNotFound(let key, let context):
                    let errorMsg = "Key '\(key.stringValue)' not found: \(context.debugDescription)"
                    throw NetworkError.serverError(errorMsg)
                    
                case .typeMismatch(let type, let context):
                    let errorMsg = "Type '\(type)' mismatch: \(context.debugDescription)"
                    throw NetworkError.serverError(errorMsg)
                    
                default:
                    throw NetworkError.invalidData
                }
            }
            
            throw NetworkError.invalidData
        }
    }
    
    func submitModelUpdate(ticker: String, date: String, actualPrice: Double) async throws -> Bool {
        guard let url = URL(string: "\(baseURL)/update_model") else {
            throw NetworkError.invalidURL
        }
        
        #if DEBUG
        print("Submitting model update to: \(url.absoluteString)")
        print("Ticker: \(ticker), Date: \(date), Price: \(actualPrice)")
        #endif
        
        // Create request body
        let body: [String: Any] = [
            "ticker": ticker,
            "date": date,
            "actual_price": actualPrice
        ]
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        
        // Perform the request
        let (data, response) = try await URLSession.shared.data(for: request)
        
        // Check response status
        guard let httpResponse = response as? HTTPURLResponse else {
            throw NetworkError.invalidResponse
        }
        
        // Handle server errors
        if httpResponse.statusCode >= 400 {
            // Try to parse error message
            if let errorObject = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let errorMessage = errorObject["error"] as? String {
                throw NetworkError.serverError(errorMessage)
            } else {
                throw NetworkError.serverError("Server returned status code \(httpResponse.statusCode)")
            }
        }
        
        return true
    }
}
