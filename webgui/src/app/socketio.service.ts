import { Injectable } from '@angular/core';
import * as io from 'socket.io-client';
import { ConfigService } from './config.service';
import { fromEvent } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class SocketioService {
  socket: io;
  constructor(private config: ConfigService) { }

  setupSocketConnection() {
    this.socket = io(this.config.serverCreds.url);
    // this.socket.emit('json', this.config);
  }

  getMessages(eventName: string) {
    return fromEvent(this.socket, eventName);
  }

  // listenOn(eventName: string, f: (data: any) => void) {
  //   return this.socket.on(eventName, f);
  // }
}
