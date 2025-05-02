// StockPrediction.swift
// Models for your stock prediction data

import Foundation

// Model to hold stock prediction data
struct StockPrediction: Decodable {
    let ticker: String
    let currentPrice: Double?
    let oneWeekPrediction: Prediction
    let oneMonthPrediction: Prediction
    let modelSummary: ModelSummary
    let dataPlot: String? // Base64 encoded image
    
    struct Prediction: Decodable {
        let price: Double?
        let changePercent: Double?
        let recommendation: String
        let confidence: Double?
    }
    
    struct ModelSummary: Decodable {
        let status: String?
        let mse: Double?
        let rmse: Double?
        let directionalAccuracy: Double?
        
        enum CodingKeys: String, CodingKey {
            case status
            case mse
            case rmse
            case directionalAccuracy = "directional_accuracy"
        }
    }
    
    enum CodingKeys: String, CodingKey {
        case ticker
        case currentPrice = "current_price"
        case oneWeekPrediction = "one_week_prediction"
        case oneMonthPrediction = "one_month_prediction"
        case modelSummary = "model_summary"
        case dataPlot = "data_plot"
    }
}
