import SwiftUI

struct BatchResultsView: View {
    let batch: PULSEBatchPresentation
    @ObservedObject var session: PULSEAppSession

    var body: some View {
        ZStack {
            PULSEAppBackground()

            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    heroSection

                    LazyVStack(spacing: 14) {
                        ForEach(batch.analyses) { analysis in
                            NavigationLink {
                                ResultsView(analysis: analysis, image: session.image(for: analysis))
                            } label: {
                                batchRow(analysis)
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 16)
            }
        }
        .navigationTitle(batch.analyses.count > 1 ? "Batch Results" : "Study Result")
        .navigationBarTitleDisplayMode(.inline)
        .toolbarBackground(.visible, for: .navigationBar)
        .toolbarBackground(Color.white.opacity(0.92), for: .navigationBar)
    }

    private var heroSection: some View {
        PULSEHeroSurface {
            VStack(alignment: .leading, spacing: 16) {
                HStack(alignment: .top, spacing: 14) {
                    VStack(alignment: .leading, spacing: 8) {
                        PULSEStatusBadge(title: "Case List", tint: .white, systemImage: "list.bullet.rectangle.portrait.fill")
                        Text(batch.title)
                            .font(.system(size: 30, weight: .bold, design: .rounded))
                            .foregroundStyle(.white)
                            .fixedSize(horizontal: false, vertical: true)
                        Text("Each study is stored locally with a structured result, visual evidence, and PDF export.")
                            .font(.subheadline)
                            .foregroundStyle(Color.white.opacity(0.82))
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    Spacer()
                    PULSEBrandMark(size: 62)
                }

                HStack(spacing: 10) {
                    PULSEMetricPill(title: "Studies", value: "\(batch.analyses.count)", icon: "square.stack.3d.up.fill")
                    PULSEMetricPill(title: "Created", value: batch.createdAt.formatted(.dateTime.hour().minute()), icon: "clock.fill")
                    PULSEMetricPill(title: "Storage", value: "Local", icon: "lock.fill")
                }
            }
        }
    }

    private func batchRow(_ analysis: PULSESavedAnalysis) -> some View {
        HStack(spacing: 14) {
            thumbnail(for: analysis)

            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 8) {
                    PULSEInfoChip(title: analysis.subtitle, systemImage: analysis.subtitle.pulseDomainSymbol)
                    Spacer()
                    PULSEStatusBadge(
                        title: analysis.report.quality.qualityLabel.capitalized,
                        tint: analysis.report.quality.tone.color,
                        systemImage: "checkmark.seal.fill"
                    )
                }

                Text(analysis.title)
                    .font(.headline.weight(.semibold))
                    .foregroundStyle(PULSEPalette.navy)
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)

                Text(analysis.displaySourceName)
                    .font(.footnote.weight(.medium))
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)

                HStack(spacing: 10) {
                    PULSEInfoChip(
                        title: (analysis.report.subviewText ?? analysis.report.detectedLabel).pulseDisplayText,
                        systemImage: analysis.report.subviewText == nil ? "checkmark.circle.fill" : "scope"
                    )
                    if let confidence = analysis.report.primaryConfidencePercentText {
                        PULSEInfoChip(title: confidence, systemImage: "percent", tint: PULSEPalette.teal)
                    }
                    Text(analysis.createdAt.formatted(.dateTime.hour().minute()))
                        .font(.caption.weight(.medium))
                        .foregroundStyle(.secondary)
                }
            }
            .layoutPriority(1)
        }
        .overlay(alignment: .trailing) {
            Image(systemName: "chevron.right.circle.fill")
                .font(.system(size: 22))
                .foregroundStyle(PULSEPalette.accent.opacity(0.9))
                .padding(.trailing, 4)
        }
        .pulseCard(background: Color.white.opacity(0.82))
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
                        .font(.system(size: 22, weight: .semibold))
                        .foregroundStyle(PULSEPalette.accent)
                }
            }
        }
        .frame(width: 82, height: 82)
        .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .stroke(Color.white.opacity(0.72), lineWidth: 1)
        )
    }
}
