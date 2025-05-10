//
//  StatefulPreview.swift
//  PlanWise
//
//  Created by Ricardo Mendez Cavalieri on 25/3/25.
//

import SwiftUI

// Preview View used to generates previews that take state variables as parameters
struct StatefulPreviewWrapper<Value, Content: View>: View {
    final class Wrapper: ObservableObject {
        @Published var value: Value
        
        init(_ value: Value) {
            self.value = value
        }
    }
    
    @ObservedObject private var wrapper: Wrapper
    
    let content: (Binding<Value>) -> Content
    
    init(_ value: Value, @ViewBuilder content: @escaping (Binding<Value>) -> Content) {
        wrapper = Wrapper(value)
        self.content = content
    }
    
    var body: some View {
        content($wrapper.value)
    }
}
