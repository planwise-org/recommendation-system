//
//  TabBarView.swift
//  PlanWise
//
//  Created by Ricardo Mendez Cavalieri on 25/3/25.
//

import SwiftUI

// Enum representing each different icon
enum Tabs: Int {
    case home = 1
    case profile = 2
}

// Tab Bar view
struct CustomTabBar: View {
    
    // Binding of type Tabs to the corresponding tab the user is in
    @Binding var selectedTab: Tabs
    
    var body: some View {
        ZStack{
            Color("White")
                .opacity(0.7)
            VStack {
                Spacer()
                tabButtons
                Spacer()
                Spacer()
            }
            }
            .frame(maxWidth: .infinity)
            .frame(height:80)
        }
        
}

struct CustomTabBar_Previews: PreviewProvider {
    static var previews: some View {
        StatefulPreviewWrapper(Tabs.home) {state in
            CustomTabBar(selectedTab: state)
        }
    }
}


extension CustomTabBar {
    
    /**
     HStack containing all different icons of the Tab Bar. selectTab and the image selected
     is modified when the button is pressed so it is showed as filled.
     */
    private var tabButtons: some View {
        HStack(spacing: 100) {
            Button {
                selectedTab = .home
            } label: {
                TabBarButton(
                    image: "map",
                    imageActive: "map.fill",
                    isActive: selectedTab == .home,
                    width: 30
                )
                .foregroundColor(Color(selectedTab == .home ? "Orange" : "Black"))
            }

            Button {
                selectedTab = .profile
            } label: {
                TabBarButton(
                    image: "person",
                    imageActive: "person.fill",
                    isActive: selectedTab == .profile,
                    width: 25
                )
                .foregroundColor(Color(selectedTab == .profile ? "Orange" : "Black"))
            }
        }
    }
}
