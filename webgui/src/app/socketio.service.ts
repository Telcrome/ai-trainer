import { Injectable } from '@angular/core';
import * as io from 'socket.io-client';
import { ConfigService } from './config.service';

@Injectable({
  providedIn: 'root'
})
export class SocketioService {
  socket: io;
  constructor(private config: ConfigService) { }

  setupSocketConnection() {
    this.socket = io(this.config.serverCreds.url);
    this.socket.emit('json', this.config);
    this.listenOn('log', (data: any) => {
      console.log(data);
    });
  }

  listenOn(eventName: string, f: (data: any) => void) {
    return this.socket.on(eventName, f)
  }
}
