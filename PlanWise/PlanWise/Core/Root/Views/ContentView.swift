//
//  ContentView.swift
//  PlanWise
//
//  Created by Ricardo Mendez Cavalieri on 25/3/25.
//

import SwiftUI

struct ContentView: View {
    
    @State private var selectedTab: Tabs = .home
    
    var body: some View {
        VStack(spacing: 0) {
            // Main content area (fills available space)
            Group {
                switch selectedTab {
                case .home:
                    VStack {
                        Image(systemName: "globe")
                            .imageScale(.large)
                            .foregroundStyle(.tint)
                        Text("Home View")
                    }
                case .profile:
                    VStack {
                        Image(systemName: "person.circle")
                            .imageScale(.large)
                            .foregroundStyle(.tint)
                        Text("Profile View")
                    }
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            
            // Custom Tab Bar fixed at bottom
            CustomTabBar(selectedTab: $selectedTab)
                .frame(maxWidth: .infinity)
        }
        .edgesIgnoringSafeArea(.bottom)
    }
}

#Preview {
    ContentView()
}
