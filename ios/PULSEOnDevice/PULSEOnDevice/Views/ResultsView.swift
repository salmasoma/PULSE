import SwiftUI
import UIKit

struct ResultsView: View {
    let analysis: PULSESavedAnalysis
    let image: UIImage?
    @State private var exportURL: ExportDocument?
    @State private var exportError: String?
    @State private var isExporting = false
    @State private var isShowingVQA = false

    var body: some View {
        ZStack {
            PULSEAppBackground()

            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    summaryHero

                    if let image {
                        imageSection(image)
                    }

                    ReportView(report: analysis.report)

                    if analysis.report.reasonedReport != nil || analysis.report.reasoningStatusMessage != nil {
                        reasonedInterpretationSection(
                            report: analysis.report.reasonedReport,
                            statusMessage: analysis.report.reasoningStatusMessage
                        )
                    }

                    if analysis.prompt.isEmpty == false {
                        promptSection
                    }

                    if let image {
                        askQuestionsSection(image: image)
                    }
                }
                .padding(.horizontal, 20)
                .padding(.top, 12)
                .padding(.bottom, 32)
            }
        }
        .navigationTitle("Analysis Report")
        .navigationBarTitleDisplayMode(.inline)
        .task(id: analysis.id) {
            if let image, PULSEMoondream2LocalExecutor.isRuntimeReady {
                _ = image
                await PULSEMoondream2LocalExecutor.prewarm()
            }
        }
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button {
                    Task {
                        await exportPDF()
                    }
                } label: {
                    if isExporting {
                        ProgressView()
                            .tint(PULSEPalette.accent)
                    } else {
                        Label("Export PDF", systemImage: "square.and.arrow.up")
                            .labelStyle(.iconOnly)
                    }
                }
            }
        }
        .toolbarBackground(.visible, for: .navigationBar)
        .toolbarBackground(Color.white.opacity(0.92), for: .navigationBar)
        .sheet(item: $exportURL) { document in
            ActivityShareSheet(items: [document.url])
        }
        .navigationDestination(isPresented: $isShowingVQA) {
            if let image {
                PULSEVQAChatView(analysis: analysis, image: image)
            }
        }
        .alert("PDF Export Failed", isPresented: Binding(
            get: { exportError != nil },
            set: { if !$0 { exportError = nil } }
        )) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(exportError ?? "The PDF could not be created.")
        }
    }

    private var summaryHero: some View {
        PULSEHeroSurface {
            VStack(alignment: .leading, spacing: 18) {
                HStack(alignment: .top, spacing: 16) {
                    VStack(alignment: .leading, spacing: 8) {
                        PULSEStatusBadge(
                            title: "Clinical Summary",
                            tint: .white,
                            systemImage: analysis.report.detectedDomain.pulseDomainSymbol
                        )

                        Text(analysis.report.detectedDomain.pulseDisplayText)
                            .font(.system(size: 30, weight: .bold, design: .rounded))
                            .foregroundStyle(.white)
                            .lineLimit(2)
                            .fixedSize(horizontal: false, vertical: true)

                        if let subview = analysis.report.subviewText {
                            VStack(alignment: .leading, spacing: 4) {
                                Text("Subview")
                                    .font(.caption.weight(.bold))
                                    .foregroundStyle(Color.white.opacity(0.74))
                                Text(subview.pulseDisplayText)
                                    .font(.title3.weight(.semibold))
                                    .foregroundStyle(PULSEPalette.mint)
                                    .fixedSize(horizontal: false, vertical: true)
                            }
                        } else {
                            Text(analysis.report.detectedLabel.pulseDisplayText)
                                .font(.title3.weight(.semibold))
                                .foregroundStyle(PULSEPalette.mint)
                                .fixedSize(horizontal: false, vertical: true)
                        }

                        Text(analysis.report.reasonedReport == nil
                             ? "A structured local interpretation with routed findings and visual evidence."
                             : "Structured local findings paired with a detailed on-device reasoning report.")
                            .font(.subheadline)
                            .foregroundStyle(Color.white.opacity(0.82))
                            .fixedSize(horizontal: false, vertical: true)
                        Text(analysis.displaySourceName)
                            .font(.footnote.weight(.medium))
                            .foregroundStyle(PULSEPalette.mint)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    Spacer()
                    VStack(alignment: .trailing, spacing: 10) {
                        if let confidence = analysis.report.primaryConfidencePercentText {
                            PULSEConfidenceBadge(value: confidence, tint: .white)
                        }
                        PULSEBrandMark(size: 58)
                    }
                }

                HStack(spacing: 10) {
                    PULSEMetricPill(title: "Domain", value: analysis.subtitle, icon: analysis.subtitle.pulseDomainSymbol)
                    PULSEMetricPill(title: analysis.report.subviewText == nil ? "Result" : "Subview", value: (analysis.report.subviewText ?? analysis.report.detectedLabel).pulseDisplayText, icon: "scope")
                    PULSEMetricPill(
                        title: analysis.report.primaryConfidencePercentText == nil ? "Quality" : "Confidence",
                        value: analysis.report.primaryConfidencePercentText ?? analysis.report.quality.qualityLabel.capitalized,
                        icon: analysis.report.primaryConfidencePercentText == nil ? "checkmark.seal.fill" : "percent"
                    )
                }

                VStack(alignment: .leading, spacing: 10) {
                    Text("Impression")
                        .font(.headline.weight(.semibold))
                        .foregroundStyle(.white)
                    Text(analysis.report.impressionText)
                        .font(.subheadline)
                        .foregroundStyle(Color.white.opacity(0.84))
                        .fixedSize(horizontal: false, vertical: true)
                }
                .pulseInsetCard(background: Color.white.opacity(0.08))

                Text(analysis.createdAt.pulseTimestamp)
                    .font(.footnote.weight(.medium))
                    .foregroundStyle(Color.white.opacity(0.78))
            }
        }
    }

    private func imageSection(_ image: UIImage) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            PULSESectionHeader(
                title: "Source Image",
                subtitle: "The original ultrasound frame reviewed by the on-device workflow.",
                systemImage: "photo.on.rectangle.angled"
            )

            Image(uiImage: image)
                .resizable()
                .scaledToFit()
                .frame(maxWidth: .infinity)
                .clipShape(RoundedRectangle(cornerRadius: 24, style: .continuous))
                .overlay(alignment: .bottomLeading) {
                    HStack(spacing: 8) {
                        PULSEStatusBadge(title: "\(Int(image.size.width)) × \(Int(image.size.height))", tint: PULSEPalette.accent, systemImage: "aspectratio")
                    }
                    .padding(14)
                }
        }
        .pulseCard(background: Color.white.opacity(0.82))
    }

    private var promptSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            PULSESectionHeader(
                title: "Clinical Context",
                subtitle: "Locally stored prompt accompanying this analysis.",
                systemImage: "text.bubble.fill"
            )
            Text(analysis.prompt)
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .pulseCard(background: Color.white.opacity(0.8))
    }

    private func askQuestionsSection(image: UIImage) -> some View {
        Button {
            isShowingVQA = true
        } label: {
            HStack(spacing: 14) {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Ask Questions")
                        .font(.headline.weight(.semibold))
                        .foregroundStyle(PULSEPalette.navy)
                    Text("Open an image-grounded chat window for follow-up VQA on this study.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer()
                Image(systemName: "arrow.right.circle.fill")
                    .font(.system(size: 26, weight: .semibold))
                    .foregroundStyle(PULSEPalette.accent)
            }
            .padding(18)
            .background(
                LinearGradient(
                    colors: [Color.white, Color(red: 0.97, green: 0.99, blue: 1.0)],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                ),
                in: RoundedRectangle(cornerRadius: 22, style: .continuous)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 22, style: .continuous)
                    .stroke(PULSEPalette.divider, lineWidth: 1)
            )
        }
        .buttonStyle(.plain)
        .pulseCard(background: Color.white.opacity(0.82))
    }

    private func reasonedInterpretationSection(report reasonedReport: PULSEReasonedReport?, statusMessage: String?) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            PULSESectionHeader(
                title: "Reasoned Interpretation",
                subtitle: "Clinician-facing reasoning generated locally from the structured findings and the bundled on-device VLM.",
                systemImage: "brain.head.profile"
            )

            VStack(alignment: .leading, spacing: 12) {
                if let reasonedReport {
                    Text(reasonedReport.output)
                        .font(.subheadline)
                        .foregroundStyle(PULSEPalette.navy)
                        .textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                } else if statusMessage == PULSEReasoningService.processingStatusMessage {
                    HStack(spacing: 12) {
                        ProgressView()
                            .tint(PULSEPalette.accent)
                        Text(statusMessage ?? PULSEReasoningService.processingStatusMessage)
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                } else if let statusMessage {
                    HStack(spacing: 10) {
                        PULSEInfoChip(title: "Unavailable", systemImage: "exclamationmark.triangle.fill", tint: PULSEPalette.warning)
                    }

                    Text(statusMessage)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .padding(18)
            .background(
                LinearGradient(
                    colors: [Color(red: 0.96, green: 0.99, blue: 1.0), Color.white],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                ),
                in: RoundedRectangle(cornerRadius: 22, style: .continuous)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 22, style: .continuous)
                    .stroke(PULSEPalette.divider, lineWidth: 1)
            )
        }
        .pulseCard(background: Color.white.opacity(0.82))
    }

    private func exportPDF() async {
        isExporting = true
        defer { isExporting = false }

        do {
            let url = try PULSEPDFExporter.export(analysis: analysis, image: image)
            exportURL = ExportDocument(url: url)
        } catch {
            exportError = error.localizedDescription
        }
    }
}

private struct ExportDocument: Identifiable {
    let id = UUID()
    let url: URL
}
