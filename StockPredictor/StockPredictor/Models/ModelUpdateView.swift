// ModelUpdateView.swift
// View for submitting model updates with actual prices

import SwiftUI

struct ModelUpdateView: View {
    @ObservedObject var viewModel: StockViewModel
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Text("Model Update")
                    .font(.headline)
                
                Spacer()
                
                Button(action: {
                    viewModel.showingUpdateForm.toggle()
                }) {
                    Label(
                        viewModel.showingUpdateForm ? "Hide" : "Show",
                        systemImage: viewModel.showingUpdateForm ? "chevron.up" : "chevron.down"
                    )
                    .font(.subheadline)
                }
            }
            
            if viewModel.showingUpdateForm {
                Text("Correct any inaccuracies in our data by submitting known stock prices.")
                    .font(.subheadline)
                    .foregroundColor(.gray)
                Text("Test What if scenarios by providing different price points")
                    .font(.subheadline)
                    .foregroundColor(.gray)
                
                Divider()
                
                // Date and price inputs
                VStack(spacing: 12) {
                    DatePicker(
                        "Select Date",
                        selection: $viewModel.updateDate,
                        in: ...Date(),  // Past dates only
                        displayedComponents: .date
                    )
                    
                    HStack {
                        Text("Actual Price:")
                        TextField("$0.00", text: $viewModel.updatePrice)
                            .keyboardType(.decimalPad)
                            .textFieldStyle(.roundedBorder)
                            .multilineTextAlignment(.trailing)
                    }
                }
                
                // Submit button
                Button(action: {
                    viewModel.submitModelUpdate()
                }) {
                    HStack {
                        Text("Submit Feedback")
                        
                        if viewModel.isUpdating {
                            Spacer()
                            ProgressView()
                                .scaleEffect(0.8)
                        }
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(10)
                }
                .disabled(viewModel.isUpdating || viewModel.updatePrice.isEmpty)
                
                // Success message
                if viewModel.updateSuccess {
                    HStack {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundColor(.green)
                        Text("Model updated successfully!")
                            .foregroundColor(.green)
                    }
                    .padding(.top, 4)
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(10)
        .animation(.default, value: viewModel.showingUpdateForm)
    }
}

struct ModelUpdateView_Previews: PreviewProvider {
    static var previews: some View {
        ModelUpdateView(viewModel: StockViewModel())
            .padding()
            .previewLayout(.sizeThatFits)
    }
}
