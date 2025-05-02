// AppDelegate.swift
// App delegate for application lifecycle events

import UIKit

class AppDelegate: NSObject, UIApplicationDelegate {
    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]? = nil) -> Bool {
        
        // Print out the documents directory for debugging
        #if DEBUG
        let documentsPath = NSSearchPathForDirectoriesInDomains(.documentDirectory, .userDomainMask, true)[0]
        print("Documents Directory: \(documentsPath)")
        #endif
        
        // Configure URLSession
        let sessionConfig = URLSessionConfiguration.default
        sessionConfig.timeoutIntervalForRequest = 60.0  // 60 second timeout
        sessionConfig.timeoutIntervalForResource = 120.0  // 2 minute overall timeout
        
        return true
    }
}
