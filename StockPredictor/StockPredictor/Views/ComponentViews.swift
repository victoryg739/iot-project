// ComponentViews.swift
// Reusable view components for the app

import SwiftUI

struct StockSummaryView: View {
    let prediction: StockPrediction
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Current Price
            HStack {
                VStack(alignment: .leading) {
                    Text(prediction.ticker)
                        .font(.largeTitle)
                        .fontWeight(.bold)
                    
                    if let currentPrice = prediction.currentPrice {
                        Text("Current Price: $\(String(format: "%.2f", currentPrice))")
                            .font(.title2)
                    }
                }
                Spacer()
            }
            
            Divider()
            
            // Predictions
            HStack(alignment: .top, spacing: 16) {
                // One week prediction
                PredictionCardView(
                    title: "1 Week Forecast",
                    prediction: prediction.oneWeekPrediction,
                    currentPrice: prediction.currentPrice
                )
                
                // One month prediction
                PredictionCardView(
                    title: "1 Month Forecast",
                    prediction: prediction.oneMonthPrediction,
                    currentPrice: prediction.currentPrice
                )
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(10)
    }
}

struct PredictionCardView: View {
    let title: String
    let prediction: StockPrediction.Prediction
    let currentPrice: Double?
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.headline)
            
            if let price = prediction.price {
                Text("$\(String(format: "%.2f", price))")
                    .font(.title3)
                    .fontWeight(.bold)
                
                if let changePercent = prediction.changePercent {
                    Text("\(changePercent >= 0 ? "+" : "")\(String(format: "%.2f", changePercent))%")
                        .foregroundColor(changePercent >= 0 ? .green : .red)
                        .fontWeight(.semibold)
                }
            }
            
            HStack {
                Text(prediction.recommendation)
                    .font(.subheadline)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(recommendationColor(prediction.recommendation))
                    .foregroundColor(.white)
                    .cornerRadius(4)
            }
            
            if let confidence = prediction.confidence {
                Text("Confidence: \(String(format: "%.0f", confidence * 100))%")
                    .font(.caption)
                    .foregroundColor(.gray)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(Color.white)
        .cornerRadius(8)
        .shadow(color: Color.black.opacity(0.1), radius: 2, x: 0, y: 1)
    }
    
    private func recommendationColor(_ recommendation: String) -> Color {
        switch recommendation {
        case "STRONG BUY":
            return Color.green
        case "BUY":
            return Color.green.opacity(0.7)
        case "WEAK BUY":
            return Color.green.opacity(0.4)
        case "HOLD":
            return Color.gray
        case "WEAK SELL":
            return Color.red.opacity(0.4)
        case "SELL":
            return Color.red.opacity(0.7)
        case "STRONG SELL":
            return Color.red
        default:
            return Color.gray
        }
    }
}

struct ModelSummaryView: View {
    let summary: StockPrediction.ModelSummary
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Model Details")
                .font(.headline)
                .padding(.bottom, 4)
            
            if let status = summary.status {
                Text("Status: \(status)")
                    .font(.subheadline)
            }
            
            if let rmse = summary.rmse {
                Text("RMSE: \(String(format: "%.4f", rmse))")
                    .font(.subheadline)
            }
            
            if let directionalAccuracy = summary.directionalAccuracy {
                Text("Directional Accuracy: \(String(format: "%.1f", directionalAccuracy))%")
                    .font(.subheadline)
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(10)
    }
}
