// ContentView.swift
// Main view for the app with model update component added

import SwiftUI

struct ContentView: View {
    @ObservedObject private var viewModel: StockViewModel
    
    init(viewModel: StockViewModel = StockViewModel()) {
        self.viewModel = viewModel
    }
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .center, spacing: 20) {
                    Image("logo")
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(width:120)
                    // Input Section
                    VStack(alignment: .leading) {
                        Text("Enter Stock Symbol")
                            .font(.headline)
                        
                        HStack {
                            TextField("Stock Symbol (e.g., AAPL)", text: $viewModel.tickerSymbol)
                                .textFieldStyle(.roundedBorder)
                                .autocapitalization(.allCharacters)
                                .disableAutocorrection(true)
                            
                            Button(action: {
                                viewModel.fetchStockPrediction()
                            }) {
                                Text("Predict")
                                    .padding(.horizontal)
                                    .padding(.vertical, 8)
                                    .background(Color.blue)
                                    .foregroundColor(.white)
                                    .cornerRadius(10)
                            }
                            .disabled(viewModel.isLoading || viewModel.tickerSymbol.isEmpty)
                        }
                        
                        Toggle("Force model retraining", isOn: $viewModel.forceRetrain)
                            .font(.subheadline)
                            .padding(.top, 4)
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(10)
                    
                    // Loading and Error
                    if viewModel.isLoading {
                        HStack {
                            Spacer()
                            ProgressView()
                                .scaleEffect(1.5)
                                .padding()
                            Spacer()
                        }
                    }
                    
                    if let errorMessage = viewModel.errorMessage {
                        Text(errorMessage)
                            .foregroundColor(.red)
                            .padding()
                            .background(Color(.systemGray6))
                            .cornerRadius(10)
                    }
                    
                    // Results Section
                    if let prediction = viewModel.stockPrediction {
                        StockSummaryView(prediction: prediction)
                        
                        // Model Update View - ADD THIS
                        ModelUpdateView(viewModel: viewModel)
                        
                        // Chart image
                        if let plotData = prediction.dataPlot,
                           let imageData = Data(base64Encoded: plotData),
                           let uiImage = UIImage(data: imageData) {
                            VStack(alignment: .leading) {
                                Text("Price Chart")
                                    .font(.headline)
                                    .padding(.bottom, 4)
                                
                                Image(uiImage: uiImage)
                                    .resizable()
                                    .scaledToFit()
                                    .cornerRadius(10)
                            }
                            .padding()
                            .background(Color(.systemGray6))
                            .cornerRadius(10)
                        }
                        
                        // Model details
                        ModelSummaryView(summary: prediction.modelSummary)
                    }
                }
                .padding()
            }
            .navigationTitle("Stock Predictor")
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
