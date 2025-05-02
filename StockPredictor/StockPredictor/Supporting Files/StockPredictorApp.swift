// StockPredictorApp.swift
// Main app entry point

import SwiftUI

@main
struct StockPredictorApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
