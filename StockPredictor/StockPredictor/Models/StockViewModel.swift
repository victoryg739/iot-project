// StockViewModel.swift
// ViewModel for managing stock prediction data

import Foundation
import SwiftUI

class StockViewModel: ObservableObject {
    // Published properties for UI binding
    @Published var tickerSymbol = "AAPL"
    @Published var forceRetrain = false
    @Published var stockPrediction: StockPrediction?
    @Published var isLoading = false
    @Published var errorMessage: String?
    
    // Services
    private let stockService: StockServiceProtocol
    
    // Dependency injection
    init(stockService: StockServiceProtocol = StockService()) {
        self.stockService = stockService
    }
        
    func fetchStockPrediction() {
        isLoading = true
        errorMessage = nil
        
        Task {
            do {
                let prediction = try await stockService.getPrediction(for: tickerSymbol, forceRetrain: forceRetrain)
                
                // Update UI on main thread
                await MainActor.run {
                    self.stockPrediction = prediction
                    self.isLoading = false
                }
            } catch let error as NetworkError {
                await handleError(error)
            } catch {
                await handleError(.serverError("An unexpected error occurred: \(error.localizedDescription)"))
            }
        }
    }
        
    @MainActor
    private func handleError(_ error: NetworkError) {
        switch error {
        case .invalidURL:
            errorMessage = "Invalid server URL"
        case .invalidResponse:
            errorMessage = "Invalid response from server"
        case .invalidData:
            errorMessage = "Couldn't decode server response"
        case .serverError(let message):
            errorMessage = message
        }
        isLoading = false
    }
}
