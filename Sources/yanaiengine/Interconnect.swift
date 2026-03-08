import Foundation
import NIO

/// A simple protocol for exchanging Tensors over the network using SwiftNIO.
public enum Interconnect {
    
    /// A channel handler that receives raw bytes and triggers a callback with a Data object.
    internal class TensorHandler: ChannelInboundHandler {
        typealias InboundIn = ByteBuffer
        let onReceived: (Data, Channel) -> Void
        
        init(onReceived: @escaping (Data, Channel) -> Void) {
            self.onReceived = onReceived
        }
        
        func channelRead(context: ChannelHandlerContext, data: NIOAny) {
            var buffer = unwrapInboundIn(data)
            if let bytes = buffer.readBytes(length: buffer.readableBytes) {
                onReceived(Data(bytes), context.channel)
            }
        }
    }
    
    /// Acts as the Master node, listening for gradient updates from workers.
    public class Server {
        private let group = MultiThreadedEventLoopGroup(numberOfThreads: 1)
        private var channel: Channel?
        private var lastWorkerChannel: Channel?
        
        public var onGradientReceived: ((Data) -> Void)?
        
        public init() {}
        
        public func start(host: String, port: Int) throws {
            let bootstrap = ServerBootstrap(group: group)
                .serverChannelOption(ChannelOptions.backlog, value: 256)
                .serverChannelOption(ChannelOptions.socketOption(.so_reuseaddr), value: 1)
                .childChannelInitializer { [weak self] channel in
                    self?.lastWorkerChannel = channel
                    return channel.pipeline.addHandler(TensorHandler(onReceived: { data, _ in
                        self?.onGradientReceived?(data)
                    }))
                }
            
            channel = try bootstrap.bind(host: host, port: port).wait()
            print("Interconnect Server started on \(host):\(port)")
        }
        
        public func sendToWorker(data: Data) {
            guard let worker = lastWorkerChannel else { 
                print("Server: No worker connected to send to.")
                return 
            }
            var buffer = worker.allocator.buffer(capacity: data.count)
            buffer.writeBytes(data)
            worker.writeAndFlush(buffer, promise: nil)
        }
        
        public func stop() {
            try? channel?.close().wait()
            try? group.syncShutdownGracefully()
        }
    }
    
    /// Acts as the Worker node, sending gradients to the Master.
    public class Client {
        private let group = MultiThreadedEventLoopGroup(numberOfThreads: 1)
        private var channel: Channel?
        
        public var onDataReceived: ((Data) -> Void)?
        
        public init() {}
        
        public func connect(host: String, port: Int) throws {
            let bootstrap = ClientBootstrap(group: group)
                .channelOption(ChannelOptions.socketOption(.so_reuseaddr), value: 1)
                .channelInitializer { [weak self] channel in
                    channel.pipeline.addHandler(TensorHandler(onReceived: { data, _ in
                        self?.onDataReceived?(data)
                    }))
                }
            channel = try bootstrap.connect(host: host, port: port).wait()
            print("Client connected to Master at \(host):\(port)")
        }
        
        public func send(data: Data) {
            guard let channel = channel else { 
                print("Client: Not connected, cannot send.")
                return 
            }
            var buffer = channel.allocator.buffer(capacity: data.count)
            buffer.writeBytes(data)
            channel.writeAndFlush(buffer, promise: nil)
        }
        
        public func stop() {
            try? channel?.close().wait()
            try? group.syncShutdownGracefully()
        }
    }
}
