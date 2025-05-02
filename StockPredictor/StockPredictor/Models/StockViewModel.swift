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
    
    // Model update properties
    @Published var updateDate = Date()
    @Published var updatePrice = ""
    @Published var isUpdating = false
    @Published var updateSuccess = false
    @Published var showingUpdateForm = false
    
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
    
    func submitModelUpdate() {
        guard let stockPrediction = stockPrediction else {
            errorMessage = "No prediction data to update"
            return
        }
        
        guard let price = Double(updatePrice), price > 0 else {
            errorMessage = "Please enter a valid positive price"
            return
        }
        
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd"
        let dateString = dateFormatter.string(from: updateDate)
        
        isUpdating = true
        errorMessage = nil
        updateSuccess = false
        
        Task {
            do {
                let success = try await stockService.submitModelUpdate(
                    ticker: stockPrediction.ticker,
                    date: dateString,
                    actualPrice: price
                )
                
                await MainActor.run {
                    self.isUpdating = false
                    self.updateSuccess = success
                    if success {
                        self.updatePrice = ""
                        // Refresh the prediction after successful update
                        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                            self.fetchStockPrediction()
                        }
                    }
                }
            } catch let error as NetworkError {
                await handleError(error)
                await MainActor.run {
                    self.isUpdating = false
                }
            } catch {
                await handleError(.serverError("An unexpected error occurred: \(error.localizedDescription)"))
                await MainActor.run {
                    self.isUpdating = false
                }
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
