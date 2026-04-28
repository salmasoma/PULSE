import SwiftUI

enum PULSEPalette {
    static let navy = Color(red: 0.06, green: 0.15, blue: 0.25)
    static let navyDeep = Color(red: 0.03, green: 0.09, blue: 0.18)
    static let accent = Color(red: 0.03, green: 0.47, blue: 0.68)
    static let accentSoft = Color(red: 0.24, green: 0.72, blue: 0.74)
    static let mint = Color(red: 0.57, green: 0.84, blue: 0.79)
    static let teal = Color(red: 0.02, green: 0.54, blue: 0.56)
    static let canvasTop = Color(red: 0.95, green: 0.97, blue: 0.99)
    static let canvasMid = Color(red: 0.97, green: 0.98, blue: 0.99)
    static let canvasBottom = Color(red: 0.99, green: 0.995, blue: 1.0)
    static let shell = Color.white.opacity(0.80)
    static let shellStrong = Color.white.opacity(0.92)
    static let shellBorder = Color.white.opacity(0.72)
    static let divider = Color.black.opacity(0.05)
    static let success = Color(red: 0.11, green: 0.55, blue: 0.39)
    static let warning = Color(red: 0.84, green: 0.57, blue: 0.18)
    static let critical = Color(red: 0.75, green: 0.29, blue: 0.26)
    static let neutral = Color(red: 0.46, green: 0.53, blue: 0.61)
}

struct PULSEAppBackground: View {
    var body: some View {
        ZStack {
            LinearGradient(
                colors: [
                    PULSEPalette.canvasTop,
                    PULSEPalette.canvasMid,
                    PULSEPalette.canvasBottom
                ],
                startPoint: .top,
                endPoint: .bottom
            )
            .ignoresSafeArea()

            Circle()
                .fill(PULSEPalette.accent.opacity(0.07))
                .frame(width: 360, height: 360)
                .blur(radius: 48)
                .offset(x: 180, y: -290)

            Circle()
                .fill(PULSEPalette.mint.opacity(0.13))
                .frame(width: 280, height: 280)
                .blur(radius: 34)
                .offset(x: -150, y: 250)
        }
    }
}

struct PULSEBrandMark: View {
    let size: CGFloat
    var animated: Bool = false

    @State private var pulse = false

    var body: some View {
        ZStack {
            Circle()
                .fill(
                    LinearGradient(
                        colors: [PULSEPalette.navy, PULSEPalette.accent],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
                .frame(width: size, height: size)
                .shadow(color: PULSEPalette.navy.opacity(0.22), radius: 22, x: 0, y: 14)

            RoundedRectangle(cornerRadius: size * 0.24, style: .continuous)
                .stroke(Color.white.opacity(0.18), lineWidth: 1)
                .frame(width: size * 0.82, height: size * 0.82)

            Circle()
                .stroke(PULSEPalette.accent.opacity(0.2), lineWidth: 14)
                .frame(width: size * (pulse ? 1.55 : 0.94), height: size * (pulse ? 1.55 : 0.94))
                .opacity(pulse ? 0.0 : 0.6)

            Image(systemName: "cross.case.fill")
                .font(.system(size: size * 0.18, weight: .bold))
                .foregroundStyle(PULSEPalette.mint)
                .offset(y: -size * 0.18)

            Image(systemName: "waveform.path.ecg")
                .font(.system(size: size * 0.34, weight: .bold))
                .foregroundStyle(.white)
                .offset(y: size * 0.08)
        }
        .onAppear {
            guard animated else { return }
            withAnimation(.easeOut(duration: 1.8).repeatForever(autoreverses: false)) {
                pulse = true
            }
        }
    }
}

struct PULSEMetricPill: View {
    let title: String
    let value: String
    var icon: String? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                if let icon {
                    Image(systemName: icon)
                        .font(.caption2.weight(.bold))
                        .foregroundStyle(PULSEPalette.accent)
                        .frame(width: 18, height: 18)
                        .background(PULSEPalette.accent.opacity(0.08), in: RoundedRectangle(cornerRadius: 6, style: .continuous))
                }
                Text(title.uppercased())
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(PULSEPalette.neutral)
                    .lineLimit(1)
                    .minimumScaleFactor(0.72)
                    .allowsTightening(true)
            }
            Text(value)
                .font(.system(size: 17, weight: .semibold, design: .rounded))
                .foregroundStyle(PULSEPalette.navy)
                .lineLimit(1)
                .minimumScaleFactor(0.76)
                .allowsTightening(true)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 12)
        .frame(maxWidth: .infinity, minHeight: 76, alignment: .leading)
        .background(Color.white.opacity(0.88), in: RoundedRectangle(cornerRadius: 18, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .stroke(PULSEPalette.divider, lineWidth: 1)
        )
    }
}

struct PULSEInfoChip: View {
    let title: String
    let systemImage: String
    var tint: Color = PULSEPalette.accent

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: systemImage)
                .font(.caption.weight(.bold))
            Text(title)
                .font(.caption.weight(.semibold))
                .lineLimit(1)
                .minimumScaleFactor(0.72)
                .allowsTightening(true)
        }
        .foregroundStyle(tint)
        .padding(.horizontal, 10)
        .padding(.vertical, 7)
        .background(tint.opacity(0.08), in: Capsule())
    }
}

struct PULSEConfidenceBadge: View {
    let value: String
    var tint: Color = PULSEPalette.accent

    var body: some View {
        Text(value)
            .font(.caption.weight(.bold))
            .foregroundStyle(tint)
            .padding(.horizontal, 10)
            .padding(.vertical, 7)
            .background(tint.opacity(0.12), in: Capsule())
    }
}

struct PULSEStatusBadge: View {
    let title: String
    let tint: Color
    var systemImage: String? = nil

    var body: some View {
        HStack(spacing: 6) {
            if let systemImage {
                Image(systemName: systemImage)
                    .font(.caption.weight(.bold))
            }
            Text(title)
                .font(.caption.weight(.bold))
                .lineLimit(1)
                .minimumScaleFactor(0.72)
                .allowsTightening(true)
        }
        .foregroundStyle(tint)
        .padding(.horizontal, 10)
        .padding(.vertical, 7)
        .background(tint.opacity(0.12), in: Capsule())
    }
}

struct PULSESectionHeader: View {
    let title: String
    let subtitle: String
    let systemImage: String

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: systemImage)
                .font(.system(size: 18, weight: .semibold))
                .foregroundStyle(PULSEPalette.accent)
                .frame(width: 34, height: 34)
                .background(PULSEPalette.accent.opacity(0.1), in: RoundedRectangle(cornerRadius: 12, style: .continuous))

            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.title3.weight(.semibold))
                    .foregroundStyle(PULSEPalette.navy)
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
                Text(subtitle)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .lineLimit(3)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer()
        }
    }
}

struct PULSEPanelBackground: ViewModifier {
    var background: Color = PULSEPalette.shell

    func body(content: Content) -> some View {
        content
            .padding(20)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 26, style: .continuous)
                    .fill(background)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 26, style: .continuous)
                    .stroke(PULSEPalette.shellBorder, lineWidth: 1)
            )
            .shadow(color: Color.black.opacity(0.055), radius: 16, x: 0, y: 10)
    }
}

extension View {
    func pulseCard(background: Color = PULSEPalette.shell) -> some View {
        modifier(PULSEPanelBackground(background: background))
    }

    func pulseInsetCard(background: Color = Color.white.opacity(0.92)) -> some View {
        padding(16)
            .background(background, in: RoundedRectangle(cornerRadius: 22, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 22, style: .continuous)
                    .stroke(PULSEPalette.divider, lineWidth: 1)
            )
    }
}

struct PULSEHeroSurface<Content: View>: View {
    let content: Content

    init(@ViewBuilder content: () -> Content) {
        self.content = content()
    }

    var body: some View {
        ZStack(alignment: .topTrailing) {
            RoundedRectangle(cornerRadius: 30, style: .continuous)
                .fill(
                    LinearGradient(
                        colors: [
                            PULSEPalette.navyDeep,
                            PULSEPalette.navy,
                            Color(red: 0.05, green: 0.36, blue: 0.57)
                        ],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )

            Circle()
                .fill(PULSEPalette.mint.opacity(0.12))
                .frame(width: 210, height: 210)
                .blur(radius: 16)
                .offset(x: 58, y: -28)

            Circle()
                .stroke(Color.white.opacity(0.08), lineWidth: 1)
                .frame(width: 156, height: 156)
                .offset(x: 24, y: -30)

            content
                .padding(24)
        }
        .overlay(
            RoundedRectangle(cornerRadius: 30, style: .continuous)
                .stroke(Color.white.opacity(0.12), lineWidth: 1)
        )
        .shadow(color: PULSEPalette.navy.opacity(0.14), radius: 24, x: 0, y: 16)
    }
}

enum PULSETone {
    case success
    case warning
    case critical
    case info
    case neutral

    var color: Color {
        switch self {
        case .success:
            return PULSEPalette.success
        case .warning:
            return PULSEPalette.warning
        case .critical:
            return PULSEPalette.critical
        case .info:
            return PULSEPalette.accent
        case .neutral:
            return PULSEPalette.neutral
        }
    }
}

extension PULSEQualitySummary {
    var tone: PULSETone {
        switch qualityLabel.lowercased() {
        case "good":
            return .success
        case "medium":
            return .warning
        case "low":
            return .critical
        default:
            return .neutral
        }
    }
}

enum PULSEFindingCategory: String {
    case routing
    case classification
    case segmentation
    case detection
    case measurement
    case multimodal
    case other

    var symbol: String {
        switch self {
        case .routing:
            return "point.topleft.down.curvedto.point.bottomright.up"
        case .classification:
            return "checklist"
        case .segmentation:
            return "square.split.2x2"
        case .detection:
            return "viewfinder"
        case .measurement:
            return "ruler"
        case .multimodal:
            return "square.stack.3d.up"
        case .other:
            return "waveform.path.ecg"
        }
    }

    var title: String {
        rawValue.capitalized
    }

    var tone: PULSETone {
        switch self {
        case .routing:
            return .info
        case .classification:
            return .success
        case .segmentation:
            return .info
        case .detection:
            return .warning
        case .measurement:
            return .critical
        case .multimodal:
            return .info
        case .other:
            return .neutral
        }
    }
}

extension PULSEFinding {
    var confidencePercentText: String? {
        guard let confidence else { return nil }
        return "\(Int((confidence * 100.0).rounded()))%"
    }

    var displayDetails: [String] {
        details.filter { !$0.lowercased().hasPrefix("confidence:") }
    }

    var category: PULSEFindingCategory {
        let task = taskID.lowercased()
        if task.contains("domain_classification") {
            return .routing
        }
        if task.contains("subview") {
            return .classification
        }
        if task.contains("multimodal") {
            return .multimodal
        }
        if task.contains("fetalclip/") || task.contains("zero_shot") {
            return .classification
        }
        if task.contains("fetalnet/head_biometry") || task.contains("fetalnet/abdominal_circumference") || task.contains("fetalnet/femur_length") {
            return .measurement
        }
        if task.contains("segmentation") {
            return .segmentation
        }
        if task.contains("detection") {
            return .detection
        }
        if task.contains("measurement") || task.contains("imt") || task.contains("hc") {
            return .measurement
        }
        if task.contains("classification") || summary.lowercased().contains("classifier") {
            return .classification
        }
        return .other
    }
}

extension PULSEAnalysisReport {
    var subviewText: String? {
        let trimmed = detectedLabel.pulseTrimmed(for: detectedDomain)
        guard trimmed.lowercased() != detectedDomain.lowercased() else {
            return nil
        }
        return trimmed
    }

    var primaryFinding: PULSEFinding? {
        specialistFindings.first ?? routingFinding
    }

    var primaryConfidencePercentText: String? {
        primaryFinding?.confidencePercentText ?? routingFinding?.confidencePercentText
    }

    var routingFinding: PULSEFinding? {
        findings.first(where: { $0.category == .routing })
    }

    var specialistFindings: [PULSEFinding] {
        findings.filter { $0.category != .routing }
    }

    var groupedFindings: [(PULSEFindingCategory, [PULSEFinding])] {
        let order: [PULSEFindingCategory] = [.classification, .segmentation, .detection, .measurement, .multimodal, .other]
        return order.compactMap { category in
            let items = specialistFindings.filter { $0.category == category }
            return items.isEmpty ? nil : (category, items)
        }
    }

    var impressionText: String {
        let specialist = specialistFindings.prefix(2).map(\.summary)
        if specialist.isEmpty {
            if let subviewText {
                return "The study is routed to the \(detectedDomain.pulseDisplayText) domain with a primary interpretation of \(subviewText.pulseDisplayText)."
            }
            return "The study is routed to the \(detectedDomain.pulseDisplayText) domain."
        }
        return specialist.joined(separator: " ")
    }
}

extension String {
    var pulseDisplayText: String {
        replacingOccurrences(of: "_", with: " ")
            .replacingOccurrences(of: "-", with: " ")
            .replacingOccurrences(of: "/", with: " ")
            .split(separator: " ")
            .map { $0.capitalized }
            .joined(separator: " ")
    }

    var pulseDomainSymbol: String {
        switch lowercased() {
        case "cardiac":
            return "heart.text.square.fill"
        case "breast":
            return "circle.hexagongrid.fill"
        case "thyroid":
            return "waveform.path"
        case "fetal":
            return "figure.and.child.holdinghands"
        case "abdominal":
            return "cross.case.circle.fill"
        case "liver":
            return "drop.circle.fill"
        case "kidney":
            return "oval.portrait.fill"
        case "pcos":
            return "circle.grid.cross.fill"
        case "carotid":
            return "arrow.up.and.down.and.arrow.left.and.right"
        default:
            return "stethoscope"
        }
    }

    func pulseTrimmed(for domain: String) -> String {
        let domainToken = domain.lowercased()
        let raw = trimmingCharacters(in: .whitespacesAndNewlines)
        let lowered = raw.lowercased()
        guard lowered.hasPrefix(domainToken + " ") else {
            return raw
        }
        return String(raw.dropFirst(domain.count + 1))
    }
}

extension Date {
    var pulseTimestamp: String {
        formatted(.dateTime.year().month(.abbreviated).day().hour().minute())
    }
}
