//
//  TabBarButton.swift
//  PlanWise
//
//  Created by Ricardo Mendez Cavalieri on 25/3/25.
//

import SwiftUI


// Component used to build a tab bar element from a given
// file image path and its corresponding filled version
struct TabBarButton: View {
    
    var image = ""
    var imageActive = ""
    var isActive = true
    var width: CGFloat = 25
    
    var body: some View {
        if isActive {
            Image(systemName: imageActive)
                .resizable()
                .scaledToFit()
                .frame(width:width)
        } else{
            Image(systemName: image)
                .resizable()
                .scaledToFit()
                .frame(width:width)
        }

    }
}

struct TabBarButton_Previews: PreviewProvider {
    static var previews: some View {
        TabBarButton()
    }
}
