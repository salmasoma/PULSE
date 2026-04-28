import SwiftUI

struct HistoryView: View {
    @ObservedObject var session: PULSEAppSession

    var body: some View {
        ZStack {
            PULSEAppBackground()

            if session.history.isEmpty {
                emptyState
                    .padding(20)
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: 18) {
                        historyHero

                        LazyVStack(spacing: 14) {
                            ForEach(session.history) { analysis in
                                NavigationLink(value: analysis) {
                                    historyRow(analysis)
                                }
                                .buttonStyle(.plain)
                            }
                        }
                    }
                    .padding(.horizontal, 20)
                    .padding(.vertical, 16)
                }
            }
        }
        .navigationTitle("Case Archive")
        .navigationBarTitleDisplayMode(.inline)
        .toolbarBackground(.visible, for: .navigationBar)
        .toolbarBackground(Color.white.opacity(0.92), for: .navigationBar)
        .navigationDestination(for: PULSESavedAnalysis.self) { analysis in
            ResultsView(analysis: analysis, image: session.image(for: analysis))
        }
    }

    private var historyHero: some View {
        VStack(alignment: .leading, spacing: 14) {
            PULSESectionHeader(
                title: "Local Case Archive",
                subtitle: "Every completed study is retained on the device with its image and structured report.",
                systemImage: "archivebox.fill"
            )

            HStack(spacing: 10) {
                PULSEMetricPill(title: "Cases", value: "\(session.history.count)", icon: "tray.full.fill")
                PULSEMetricPill(title: "Storage", value: "Local", icon: "internaldrive.fill")
                PULSEMetricPill(title: "Privacy", value: "On Device", icon: "lock.fill")
            }
        }
        .pulseCard(background: PULSEPalette.shellStrong)
    }

    private var emptyState: some View {
        VStack(alignment: .leading, spacing: 14) {
            PULSESectionHeader(
                title: "No Saved Analyses Yet",
                subtitle: "Completed on-device studies will appear here with their input image and structured report.",
                systemImage: "tray.fill"
            )
        }
        .pulseCard()
    }

    private func historyRow(_ analysis: PULSESavedAnalysis) -> some View {
        HStack(spacing: 14) {
            thumbnail(for: analysis)

            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 8) {
                    Image(systemName: analysis.subtitle.pulseDomainSymbol)
                        .font(.caption.weight(.bold))
                        .foregroundStyle(PULSEPalette.accent)
                    Text(analysis.subtitle)
                        .font(.caption.weight(.bold))
                        .foregroundStyle(PULSEPalette.accent)
                        .lineLimit(2)
                        .fixedSize(horizontal: false, vertical: true)
                }

                Text(analysis.title)
                    .font(.headline.weight(.semibold))
                    .foregroundStyle(PULSEPalette.navy)
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
                Text(analysis.displaySourceName)
                    .font(.footnote.weight(.medium))
                    .foregroundStyle(PULSEPalette.accent)
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
                Text(analysis.createdAt.pulseTimestamp)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
            .layoutPriority(1)

            Spacer()

            VStack(alignment: .trailing, spacing: 8) {
                PULSEStatusBadge(
                    title: analysis.report.quality.qualityLabel.capitalized,
                    tint: analysis.report.quality.tone.color,
                    systemImage: "checkmark.seal.fill"
                )
                Image(systemName: "chevron.right")
                    .font(.caption.weight(.bold))
                    .foregroundStyle(PULSEPalette.accent)
            }
        }
        .pulseCard(background: Color.white.opacity(0.8))
    }

    private func thumbnail(for analysis: PULSESavedAnalysis) -> some View {
        Group {
            if let image = session.image(for: analysis) {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFill()
            } else {
                ZStack {
                    RoundedRectangle(cornerRadius: 18, style: .continuous)
                        .fill(Color.white)
                    Image(systemName: analysis.subtitle.pulseDomainSymbol)
                        .font(.system(size: 24, weight: .semibold))
                        .foregroundStyle(PULSEPalette.accent)
                }
            }
        }
        .frame(width: 82, height: 82)
        .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
    }
}
