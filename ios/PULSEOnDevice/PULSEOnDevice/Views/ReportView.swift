import Foundation
import SwiftUI

struct ReportView: View {
    let report: PULSEAnalysisReport
    @State private var selectedArtifact: PULSEArtifactImage?

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            headerSection
            triagePanel
            routingPanel
            specialistPanels
            interpretationPanel
        }
        .pulseCard(background: PULSEPalette.shellStrong)
        .fullScreenCover(item: $selectedArtifact) { artifact in
            ArtifactDetailView(artifact: artifact)
        }
    }

    private var headerSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Structured Findings")
                        .font(.title2.weight(.bold))
                        .foregroundStyle(PULSEPalette.navy)
                    Text("A concise review surface with the selected domain, the primary interpretation, and visual evidence.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer()
                domainBadge
            }

            HStack(spacing: 10) {
                PULSEInfoChip(title: report.detectedDomain.pulseDisplayText, systemImage: report.detectedDomain.pulseDomainSymbol)
                if let subview = report.subviewText {
                    PULSEInfoChip(title: subview.pulseDisplayText, systemImage: "scope", tint: PULSEPalette.teal)
                }
                if let confidence = report.primaryConfidencePercentText {
                    PULSEInfoChip(title: confidence, systemImage: "percent", tint: PULSEPalette.accent)
                }
                PULSEInfoChip(title: report.quality.qualityLabel.capitalized, systemImage: "checkmark.seal.fill", tint: report.quality.tone.color)
            }
            .padding(.top, 2)
        }
    }

    private var domainBadge: some View {
        HStack(spacing: 8) {
            Image(systemName: report.detectedDomain.pulseDomainSymbol)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(PULSEPalette.accent)
            Text(report.detectedDomain.pulseDisplayText.uppercased())
                .font(.caption.weight(.bold))
                .foregroundStyle(PULSEPalette.accent)
                .lineLimit(2)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(
            LinearGradient(
                colors: [Color(red: 0.92, green: 0.96, blue: 1.0), Color.white],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            ),
            in: RoundedRectangle(cornerRadius: 16, style: .continuous)
        )
    }

    private var triagePanel: some View {
        VStack(alignment: .leading, spacing: 14) {
            PULSESectionHeader(
                title: "AI Triage",
                subtitle: "Input quality indicators derived from the acquired frame.",
                systemImage: "waveform.badge.magnifyingglass"
            )

            HStack(spacing: 12) {
                metricTile(title: "Quality", value: report.quality.qualityLabel.capitalized, icon: "checkmark.seal.fill", tint: report.quality.tone.color)
                metricTile(title: "Brightness", value: format(report.quality.brightness), icon: "sun.max.fill", tint: PULSEPalette.warning)
                metricTile(title: "Contrast", value: format(report.quality.contrast), icon: "circle.lefthalf.filled", tint: PULSEPalette.accent)
            }
        }
        .pulseInsetCard(background: Color.white.opacity(0.86))
    }

    private func metricTile(title: String, value: String, icon: String, tint: Color) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack(spacing: 6) {
                Image(systemName: icon)
                    .font(.caption2.weight(.bold))
                    .foregroundStyle(tint)
                Text(title.uppercased())
                    .font(.caption2.weight(.bold))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .minimumScaleFactor(0.72)
                    .allowsTightening(true)
            }
            Text(value)
                .font(.headline.weight(.semibold))
                .foregroundStyle(PULSEPalette.navy)
                .lineLimit(1)
                .minimumScaleFactor(0.76)
                .allowsTightening(true)
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            LinearGradient(
                colors: [Color.white, Color(red: 0.97, green: 0.98, blue: 0.99)],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            ),
            in: RoundedRectangle(cornerRadius: 18, style: .continuous)
        )
    }

    private var routingPanel: some View {
        Group {
            if let routing = report.routingFinding {
                VStack(alignment: .leading, spacing: 14) {
                    PULSESectionHeader(
                        title: "Domain Selection",
                        subtitle: "Primary domain selection used to activate the relevant local workflow.",
                        systemImage: "point.topleft.down.curvedto.point.bottomright.up"
                    )
                    findingCard(routing, emphasized: true)
                }
            }
        }
    }

    private var specialistPanels: some View {
        VStack(alignment: .leading, spacing: 16) {
            PULSESectionHeader(
                title: "Specialist Findings",
                subtitle: "Grouped into a small set of output types for faster review.",
                systemImage: "cross.vial.fill"
            )

            ForEach(report.groupedFindings, id: \.0.rawValue) { category, findings in
                VStack(alignment: .leading, spacing: 12) {
                    HStack {
                        PULSEStatusBadge(title: category.title, tint: category.tone.color, systemImage: category.symbol)
                        Spacer()
                        Text("\(findings.count)")
                            .font(.footnote.weight(.semibold))
                            .foregroundStyle(category.tone.color)
                    }
                    ForEach(findings) { finding in
                        findingCard(finding, emphasized: false)
                    }
                }
                .pulseInsetCard(background: Color.white.opacity(0.88))
            }
        }
    }

    private var interpretationPanel: some View {
        VStack(alignment: .leading, spacing: 12) {
            PULSESectionHeader(
                title: "Interpretation",
                subtitle: "A high-level synthesis of the local findings.",
                systemImage: "doc.text.magnifyingglass"
            )

            VStack(alignment: .leading, spacing: 10) {
                Text("Impression")
                    .font(.headline.weight(.semibold))
                    .foregroundStyle(PULSEPalette.navy)
                Text(report.impressionText)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)

                Divider()
                    .overlay(PULSEPalette.divider)

                Text(report.note)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
            .padding(16)
            .background(
                LinearGradient(
                    colors: [Color(red: 0.97, green: 0.99, blue: 1.0), Color.white],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                ),
                in: RoundedRectangle(cornerRadius: 22, style: .continuous)
            )
        }
    }

    private func findingCard(_ finding: PULSEFinding, emphasized: Bool) -> some View {
        let tone = finding.category.tone.color
        return VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .top) {
                HStack(spacing: 10) {
                    Image(systemName: finding.category.symbol)
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(tone)
                        .frame(width: 34, height: 34)
                        .background(tone.opacity(0.12), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                VStack(alignment: .leading, spacing: 3) {
                    Text(finding.title)
                        .font(.headline.weight(.semibold))
                        .foregroundStyle(PULSEPalette.navy)
                        .lineLimit(2)
                        .fixedSize(horizontal: false, vertical: true)
                    HStack(spacing: 8) {
                        Text(finding.category.title.uppercased())
                            .font(.caption.weight(.bold))
                            .foregroundStyle(tone)
                        if let label = finding.label {
                            Text(label.pulseDisplayText)
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(PULSEPalette.neutral)
                        }
                    }
                }
                }
                Spacer()
                if let confidence = finding.confidencePercentText {
                    PULSEConfidenceBadge(value: confidence, tint: tone)
                } else if emphasized {
                    PULSEStatusBadge(title: "Primary", tint: tone)
                }
            }
            Text(finding.summary)
                .font(.subheadline)
                .foregroundStyle(.primary)
                .fixedSize(horizontal: false, vertical: true)
            if !finding.displayDetails.isEmpty {
                VStack(alignment: .leading, spacing: 6) {
                    ForEach(finding.displayDetails, id: \.self) { line in
                        Text(line)
                            .font(.footnote.weight(.medium))
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.black.opacity(0.03), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            }

            if !finding.artifacts.isEmpty {
                Text("Visual Evidence")
                    .font(.caption.weight(.bold))
                    .foregroundStyle(.secondary)
                artifactGallery(finding.artifacts)
            }
        }
        .padding(15)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            emphasized
                ? tone.opacity(0.07)
                : Color.white.opacity(0.86),
            in: RoundedRectangle(cornerRadius: 20, style: .continuous)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 20, style: .continuous)
                .stroke(emphasized ? tone.opacity(0.18) : PULSEPalette.divider, lineWidth: 1)
        )
    }

    private func artifactGallery(_ artifacts: [PULSEArtifactImage]) -> some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 12) {
                ForEach(artifacts) { artifact in
                    Button {
                        selectedArtifact = artifact
                    } label: {
                        VStack(alignment: .leading, spacing: 8) {
                            if let image = UIImage(data: artifact.pngData) {
                                Image(uiImage: image)
                                    .resizable()
                                    .scaledToFit()
                                    .frame(width: 204, height: 162)
                                    .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                            }
                            HStack {
                                Text(artifact.caption)
                                    .font(.caption.weight(.semibold))
                                    .foregroundStyle(PULSEPalette.navy)
                                    .lineLimit(2)
                                    .fixedSize(horizontal: false, vertical: true)
                                Spacer()
                                Image(systemName: "plus.magnifyingglass")
                                    .font(.caption.weight(.bold))
                                    .foregroundStyle(PULSEPalette.accent)
                            }
                        }
                        .padding(10)
                        .background(Color.white.opacity(0.92), in: RoundedRectangle(cornerRadius: 18, style: .continuous))
                        .overlay(
                            RoundedRectangle(cornerRadius: 18, style: .continuous)
                                .stroke(PULSEPalette.divider, lineWidth: 1)
                        )
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }

    private func format(_ value: Double) -> String {
        String(format: "%.3f", value)
    }
}

private struct ArtifactDetailView: View {
    let artifact: PULSEArtifactImage
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ZStack {
                LinearGradient(
                    colors: [
                        Color(red: 0.04, green: 0.09, blue: 0.16),
                        Color(red: 0.06, green: 0.15, blue: 0.24)
                    ],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
                .ignoresSafeArea()

                if let image = UIImage(data: artifact.pngData) {
                    ZoomableImageView(image: image)
                        .ignoresSafeArea(edges: .bottom)
                }
            }
            .navigationTitle(artifact.caption)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }
}

private struct ZoomableImageView: UIViewRepresentable {
    let image: UIImage

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    func makeUIView(context: Context) -> UIScrollView {
        let scrollView = UIScrollView()
        scrollView.delegate = context.coordinator
        scrollView.minimumZoomScale = 1.0
        scrollView.maximumZoomScale = 6.0
        scrollView.bouncesZoom = true
        scrollView.showsHorizontalScrollIndicator = false
        scrollView.showsVerticalScrollIndicator = false
        scrollView.backgroundColor = .clear

        let imageView = UIImageView(image: image)
        imageView.contentMode = .scaleAspectFit
        imageView.translatesAutoresizingMaskIntoConstraints = false
        scrollView.addSubview(imageView)
        context.coordinator.imageView = imageView

        NSLayoutConstraint.activate([
            imageView.leadingAnchor.constraint(equalTo: scrollView.contentLayoutGuide.leadingAnchor),
            imageView.trailingAnchor.constraint(equalTo: scrollView.contentLayoutGuide.trailingAnchor),
            imageView.topAnchor.constraint(equalTo: scrollView.contentLayoutGuide.topAnchor),
            imageView.bottomAnchor.constraint(equalTo: scrollView.contentLayoutGuide.bottomAnchor),
            imageView.widthAnchor.constraint(equalTo: scrollView.frameLayoutGuide.widthAnchor),
            imageView.heightAnchor.constraint(equalTo: scrollView.frameLayoutGuide.heightAnchor)
        ])
        return scrollView
    }

    func updateUIView(_ scrollView: UIScrollView, context: Context) {
        context.coordinator.imageView?.image = image
    }

    final class Coordinator: NSObject, UIScrollViewDelegate {
        var imageView: UIImageView?

        func viewForZooming(in scrollView: UIScrollView) -> UIView? {
            imageView
        }
    }
}
