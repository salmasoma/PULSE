import SwiftUI

struct SplashView: View {
    let onFinished: () -> Void

    @State private var reveal = false
    @State private var started = false

    var body: some View {
        ZStack {
            PULSEAppBackground()

            VStack(spacing: 28) {
                PULSEBrandMark(size: 112, animated: true)
                    .scaleEffect(reveal ? 1.0 : 0.82)
                    .opacity(reveal ? 1.0 : 0.0)

                VStack(spacing: 10) {
                    Text("PULSE")
                        .font(.system(size: 48, weight: .heavy, design: .rounded))
                        .tracking(1.2)
                    Text("Point-of-care Ultrasound via Lightweight Scalable Embedded TinyML")
                        .font(.headline.weight(.medium))
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                        .frame(maxWidth: 320)
                }
                .offset(y: reveal ? 0 : 20)
                .opacity(reveal ? 1.0 : 0.0)

                VStack(spacing: 10) {
                    Text("On-device ultrasound triage and specialist inference")
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(PULSEPalette.accent)
                        .padding(.horizontal, 18)
                        .padding(.vertical, 10)
                        .background(Color.white.opacity(0.72), in: Capsule())

                    Text("Designed for a clean clinical workflow")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
                .opacity(reveal ? 1.0 : 0.0)
            }
            .padding(24)
        }
        .task {
            guard !started else { return }
            started = true
            withAnimation(.spring(response: 1.0, dampingFraction: 0.84)) {
                reveal = true
            }
            try? await Task.sleep(for: .seconds(2.4))
            onFinished()
        }
    }
}
