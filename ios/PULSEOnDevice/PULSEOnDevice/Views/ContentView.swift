import SwiftUI

struct ContentView: View {
    @StateObject private var session = PULSEAppSession()

    var body: some View {
        ZStack {
            if session.hasCompletedSplash {
                MainTabShell(session: session)
                    .transition(.asymmetric(insertion: .opacity.combined(with: .move(edge: .bottom)), removal: .opacity))
            } else {
                SplashView {
                    session.completeSplash()
                }
                .transition(.opacity)
            }
        }
        .preferredColorScheme(.light)
    }
}

private struct MainTabShell: View {
    @ObservedObject var session: PULSEAppSession

    var body: some View {
        TabView(selection: $session.selectedTab) {
            NavigationStack {
                HomeView(session: session, pipeline: session.pipeline)
            }
            .tabItem {
                Label("Home", systemImage: "house.fill")
            }
            .tag(PULSEAppSession.Tab.home)

            NavigationStack {
                HistoryView(session: session)
            }
            .tabItem {
                Label("History", systemImage: "clock.arrow.trianglehead.counterclockwise.rotate.90")
            }
            .tag(PULSEAppSession.Tab.history)
        }
        .tint(PULSEPalette.accent)
        .toolbarBackground(.visible, for: .tabBar)
        .toolbarBackground(Color.white.opacity(0.96), for: .tabBar)
    }
}
