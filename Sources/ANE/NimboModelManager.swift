#if os(iOS)
import CoreML
import Foundation
import NimboCore

@MainActor @Observable
final class NimboModelManager {
    enum State: Equatable {
        case idle
        case loading(progress: Double, stage: String)
        case ready
        case error(String)

        static func == (lhs: State, rhs: State) -> Bool {
            switch (lhs, rhs) {
            case (.idle, .idle), (.ready, .ready):
                true
            case let (.loading(lp, ls), .loading(rp, rs)):
                lp == rp && ls == rs
            case let (.error(l), .error(r)):
                l == r
            default:
                false
            }
        }
    }

    private(set) var state: State = .idle
    private(set) var inferenceManager: InferenceManager?
    private(set) var tokenizer: NimboCore.Tokenizer?
    private(set) var config: YAMLConfig?
    private(set) var loadedDirectoryPath: String?

    private var loadTask: Task<Void, Never>?

    var isReady: Bool {
        if case .ready = self.state { return true }
        return false
    }

    var loadingProgress: Double? {
        if case let .loading(progress, _) = self.state { return progress }
        return nil
    }

    var loadingStage: String? {
        if case let .loading(_, stage) = self.state { return stage }
        return nil
    }

    var errorMessage: String? {
        if case let .error(msg) = self.state { return msg }
        return nil
    }

    func loadModel(from directoryPath: String) {
        self.loadTask?.cancel()
        self.unloadModel()

        self.state = .loading(progress: 0, stage: "Reading configuration…")

        self.loadTask = Task { [weak self] in
            guard let self else { return }
            do {
                let metaPath = (directoryPath as NSString).appendingPathComponent("meta.yaml")
                let yamlConfig = try YAMLConfig.load(from: metaPath)

                await MainActor.run {
                    self.config = yamlConfig
                    self.state = .loading(progress: 0.05, stage: "Loading tokenizer…")
                }

                let tok = try await NimboCore.Tokenizer(
                    modelPath: directoryPath,
                    template: yamlConfig.modelPrefix,
                    debugLevel: 0)

                guard !Task.isCancelled else { return }

                await MainActor.run {
                    self.state = .loading(progress: 0.1, stage: "Loading CoreML models…")
                }

                let loader = ModelLoader()
                let loaderConfig = ModelLoader.Configuration(
                    computeUnits: .cpuAndNeuralEngine,
                    functionName: yamlConfig.functionName)
                let loadedModels = try await loader.loadModel(
                    from: yamlConfig,
                    configuration: loaderConfig)

                guard !Task.isCancelled else { return }

                await MainActor.run {
                    self.state = .loading(progress: 0.9, stage: "Initializing inference engine…")
                }

                let mgr = try InferenceManager(
                    models: loadedModels,
                    contextLength: yamlConfig.contextLength,
                    batchSize: yamlConfig.batchSize,
                    splitLMHead: yamlConfig.splitLMHead,
                    debugLevel: 0,
                    argmaxInModel: yamlConfig.argmaxInModel,
                    slidingWindow: yamlConfig.slidingWindow,
                    updateMaskPrefill: yamlConfig.updateMaskPrefill,
                    prefillDynamicSlice: yamlConfig.prefillDynamicSlice,
                    modelPrefix: yamlConfig.modelPrefix,
                    vocabSize: yamlConfig.vocabSize,
                    lmHeadChunkSizes: yamlConfig.lmHeadChunkSizes)

                try mgr.initializeBackings()
                mgr.initFullCausalMask()
                mgr.initState()

                if let sampling = yamlConfig.recommendedSampling {
                    mgr.setSamplingConfig(SamplingConfig(
                        doSample: sampling.doSample,
                        temperature: sampling.temperature,
                        topK: sampling.topK,
                        topP: sampling.topP,
                        repetitionPenalty: 1.1))
                } else {
                    mgr.setSamplingConfig(.defaultSampling)
                }

                guard !Task.isCancelled else { return }

                await MainActor.run {
                    self.inferenceManager = mgr
                    self.tokenizer = tok
                    self.loadedDirectoryPath = directoryPath
                    self.state = .ready
                }
            } catch {
                guard !Task.isCancelled else { return }
                await MainActor.run {
                    self.state = .error(error.localizedDescription)
                }
            }
        }
    }

    func unloadModel() {
        self.loadTask?.cancel()
        self.loadTask = nil
        self.inferenceManager?.unload()
        self.inferenceManager = nil
        self.tokenizer = nil
        self.config = nil
        self.loadedDirectoryPath = nil
        self.state = .idle
    }

    /// List model directories in the app's Documents/models folder.
    /// Each directory must contain a meta.yaml file.
    static func availableModelDirectories() -> [URL] {
        let fileManager = FileManager.default
        guard let documentsURL = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first else {
            return []
        }
        let modelsURL = documentsURL.appendingPathComponent("models", isDirectory: true)
        guard let contents = try? fileManager.contentsOfDirectory(
            at: modelsURL,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles])
        else {
            return []
        }
        return contents.filter { url in
            var isDir: ObjCBool = false
            guard fileManager.fileExists(atPath: url.path, isDirectory: &isDir), isDir.boolValue else {
                return false
            }
            let metaPath = url.appendingPathComponent("meta.yaml")
            return fileManager.fileExists(atPath: metaPath.path)
        }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }
}
#endif
